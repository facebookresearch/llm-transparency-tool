import argparse
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd
import plotly.express
import plotly.graph_objects
import streamlit as st
import streamlit_extras.row as st_row
import torch
from jaxtyping import Float
from pyinstrument import Profiler
from torch.amp import autocast
from transformers import HfArgumentParser
from typeguard import typechecked

import llm_transparency_tool.components
import llm_transparency_tool.routes.contributions as contributions
import llm_transparency_tool.routes.graph
from llm_transparency_tool.models.transparent_llm import TransparentLlm
from llm_transparency_tool.routes.graph_node import NodeType
from llm_transparency_tool.server.graph_selection import (
    GraphSelection,
    UiGraphEdge,
    UiGraphNode,
)
from llm_transparency_tool.server.styles import (
    RenderSettings,
    logits_color_map,
    margins_css,
    string_to_display,
)
from llm_transparency_tool.server.utils import (
    B0,
    GPU,
    get_contribution_graph,
    init_gpu_memory,
    load_dataset,
    load_model_with_session_caching,
    possible_devices,
    run_model_with_session_caching,
)


@dataclass
class LlmViewerConfig:
    debug: bool = field(
        default=False,
        metadata={"help": "Show debugging information, like the time profile."},
    )

    preloaded_dataset_filename: str = field(
        default=None,
        metadata={"help": "The name of the text file to load the lines from."},
    )

    allow_loading_dataset_files: bool = field(
        default=True,
        metadata={
            "help": "Whether the app should be able to load the dataset files "
            "on the server side."
        },
    )

    max_user_string_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "Limit for the length of user-provided sentences (in characters), "
            "or None if there is no limit."
        },
    )

    models: Dict[str, str] = field(
        default_factory=dict,
        metadata={
            "help": "Locations of models which are stored locally. Dictionary: official "
            "HuggingFace name -> path to dir. If None is specified, the model will be"
            "downloaded from HuggingFace."
        },
    )

    default_model: str = field(
        default="",
        metadata={"help": "The model to load once the UI is started."},
    )


class UnembeddingMode(Enum):
    """
    Determines how the intermediate representations are projected onto the vocabulary.
    """

    # Apply output layernorm + output unembedding matrix. This is what the model is
    # supposed to do with the last layer.
    NORMALIZE_AND_UNEMBED = "Normalize and unembed"

    # Apply the output unembedding matrix only. This mode can be potentially removed if
    # proven to be useless.
    UNEMBED = "Unembed"


class ThresholdMode(Enum):
    """
    Determines thresholding rules for the contribution graph.
    """

    # A mode that was used for the original graphs paper. The threshold is applied to
    # the raw contributions. Then contributions are renormalized in order to make them
    # sum up to 1 again.
    RENORMALIZE = "Cut and renormalize"

    # In this mode the threshold is applied directly to the edges of the graph during
    # the traverse. It's more straightforward and more efficient computationally because
    # we can cache contributions as they are.
    PLAIN = "Cut only"


class App:
    _model: Optional[TransparentLlm] = None
    _settings = RenderSettings()
    _graph: Optional[nx.Graph] = None
    _unembedding_mode = UnembeddingMode.NORMALIZE_AND_UNEMBED
    _threshold_mode = ThresholdMode.RENORMALIZE
    _contribution_threshold: float = 0.0

    def __init__(self, config: LlmViewerConfig):
        self._config = config
        self.gpu_memory_overhead = None

    def _get_representation(
        self, node: Optional[UiGraphNode]
    ) -> Optional[Float[torch.Tensor, "d_model"]]:
        if node is None:
            return None
        fn = {
            NodeType.AFTER_ATTN: self._model.residual_after_attn,
            NodeType.AFTER_FFN: self._model.residual_out,
            NodeType.FFN: None,
            NodeType.ORIGINAL: self._model.residual_in,
        }
        return fn[node.type](node.layer)[B0][node.token]

    def prepare_page(self):
        st.set_page_config(layout="wide")
        st.markdown(margins_css, unsafe_allow_html=True)

    def draw_model_info(self) -> None:
        info = self._model.model_info().__dict__
        df = pd.DataFrame(
            data=[str(x) for x in info.values()],
            index=info.keys(),
            columns=["Model parameter"],
        )
        st.dataframe(df, use_container_width=False)

    def draw_dataset_selection(self) -> int:
        def update_dataset(filename: Optional[str]):
            dataset = load_dataset(filename) if filename is not None else []
            st.session_state["dataset"] = dataset
            st.session_state["dataset_file"] = filename

        if "dataset" not in st.session_state:
            update_dataset(self._config.preloaded_dataset_filename)

        if self._config.allow_loading_dataset_files:
            row_f = st_row.row([2, 1], vertical_align="bottom")
            filename = row_f.text_input(
                "Dataset", value=st.session_state.dataset_file or ""
            )
            if row_f.button("Load"):
                update_dataset(filename)

        row_s = st_row.row([2, 1], vertical_align="bottom")
        new_sentence = row_s.text_input("New sentence")
        new_sentence_added = False
        if row_s.button("Add"):
            max_len = self._config.max_user_string_length
            n = len(new_sentence)
            if max_len is None or n <= max_len:
                st.session_state.dataset.append(new_sentence)
                new_sentence_added = True
            else:
                st.warning(
                    f"Sentence length {n} is larger than "
                    f"the configured limit of {max_len}"
                )

        sentences = st.session_state.dataset
        selection = st.selectbox(
            "Sentence",
            sentences,
            index=(
                len(sentences) - 1
                if new_sentence_added
                else sentences.index(
                    st.session_state.get("sentence_selector", sentences[0])
                )
            ),
            key="sentence_selector",
        )
        return selection

    def draw_threshold_slider(self) -> float:
        return st.slider(
            min_value=0.0,
            max_value=0.1,
            step=0.005,
            value=0.04,
            format=r"%.3f",
            label="Contribution threshold",
        )

    def draw_unembedding_mode(self) -> UnembeddingMode:
        modes = [m.value for m in UnembeddingMode]
        selection = st.selectbox(
            "Unembedding mode",
            modes,
            index=modes.index(UnembeddingMode.NORMALIZE_AND_UNEMBED.value),
        )
        return UnembeddingMode(selection)

    def draw_threshold_mode(self) -> ThresholdMode:
        modes = [m.value for m in ThresholdMode]
        selection = st.selectbox(
            "Threshold mode",
            modes,
            # TODO(igortufanov): Change the default to "PLAIN" after the graphs paper is
            # done. Plain threshold is faster and more straightforward.
            index=modes.index(ThresholdMode.RENORMALIZE.value),
        )
        return ThresholdMode(selection)

    @typechecked
    def _unembed(
        self, representation: Float[torch.Tensor, "d_model"]
    ) -> Float[torch.Tensor, "d_vocab"]:
        if self._unembedding_mode == UnembeddingMode.UNEMBED:
            return self._model.unembed(representation, normalize=False)
        elif self._unembedding_mode == UnembeddingMode.NORMALIZE_AND_UNEMBED:
            return self._model.unembed(representation, normalize=True)
        else:
            raise RuntimeError(f"Unknown unembedding mode {self._unembedding_mode}")

    def draw_gpu_memory(self):
        if not torch.cuda.is_available():
            return

        st.write("GPU memory on server")
        for i in range(torch.cuda.device_count()):
            [free, total] = [x / 1024 / 1024 for x in torch.cuda.mem_get_info(i)]
            occupied = total - free
            st.progress(
                self.gpu_memory_overhead[i] / total,
                f"GPU{i} overhead {self.gpu_memory_overhead[i]:.0f}M / {total:.0f}M",
            )
            st.progress(occupied / total, f"GPU{i} {occupied:.0f}M / {total:.0f}M")
            st.metric("MB", occupied - self.gpu_memory_overhead[i])

    def draw_graph(self, contribution_threshold: float) -> Optional[GraphSelection]:
        tokens = self._model.tokens()[B0]
        n_tokens = tokens.shape[0]
        model_info = self._model.model_info()

        graphs = llm_transparency_tool.routes.graph.build_paths_to_predictions(
            self._graph,
            model_info.n_layers,
            n_tokens,
            range(n_tokens),
            contribution_threshold,
        )

        return llm_transparency_tool.components.contribution_graph(
            model_info,
            self._model.tokens_to_strings(tokens),
            graphs,
            key="graph",
        )

    @typechecked
    def draw_token_matrix(
        self,
        values: Float[torch.Tensor, "t t"],
        tokens: List[str],
        value_name: str,
        title: str,
    ):
        assert values.shape[0] == len(tokens)
        labels = {
            "x": "src",
            "y": "tgt",
            "color": value_name,
        }

        captions = [f"({i}){t}" for i, t in enumerate(tokens)]

        fig = plotly.express.imshow(
            values.cpu(),
            title=title,
            labels=labels,
            x=captions,
            y=captions,
            color_continuous_scale=self._settings.attention_color_map,
            aspect="equal",
        )
        fig.update_xaxes(tickmode="linear")
        fig.update_yaxes(tickmode="linear")
        fig.update_coloraxes(showscale=False)

        st.plotly_chart(fig, use_container_width=False, theme=None)

    def draw_attn_info(
        self, edge: UiGraphEdge, container_attention_map
    ) -> Optional[int]:
        """
        Returns: the index of the selected head.
        """

        n_heads = self._model.model_info().n_heads

        layer = edge.target.layer

        head_contrib, _ = contributions.get_attention_contributions(
            resid_pre=self._model.residual_in(layer)[B0].unsqueeze(0),
            resid_mid=self._model.residual_after_attn(layer)[B0].unsqueeze(0),
            decomposed_attn=self._model.decomposed_attn(B0, layer).unsqueeze(0),
        )

        # [batch pos key_pos head] -> [head]
        flat_contrib = head_contrib[0, edge.target.token, edge.source.token, :]
        assert flat_contrib.shape[0] == n_heads, f"{flat_contrib.shape} vs {n_heads}"

        selected_head = llm_transparency_tool.components.selector(
            items=[f"H{h}" if h >= 0 else "All" for h in range(-1, n_heads)],
            indices=range(-1, n_heads),
            temperatures=[sum(flat_contrib).item()] + flat_contrib.tolist(),
            preselected_index=flat_contrib.argmax().item(),
            key="head_selector",
        )
        if selected_head == -1:
            selected_head = None

        # Draw attention matrix and contributions for the selected head.
        if selected_head is not None:
            tokens = [
                string_to_display(s)
                for s in self._model.tokens_to_strings(self._model.tokens()[B0])
            ]

            with container_attention_map:
                attn_container, contrib_container = st.columns([1, 1])
                with attn_container:
                    attn = self._model.attention_matrix(B0, layer, selected_head)
                    self.draw_token_matrix(
                        attn,
                        tokens,
                        "attention",
                        f"Attention map L{layer} H{selected_head}",
                    )
                with contrib_container:
                    contrib = head_contrib[B0, :, :, selected_head]
                    self.draw_token_matrix(
                        contrib,
                        tokens,
                        "contribution",
                        f"Contribution map L{layer} H{selected_head}",
                    )

        return selected_head

    def draw_ffn_info(self, node: UiGraphNode) -> Optional[int]:
        """
        Returns: the index of the selected neuron.
        """

        resid_mid = self._model.residual_after_attn(node.layer)[B0][node.token]
        resid_post = self._model.residual_out(node.layer)[B0][node.token]
        decomposed_ffn = self._model.decomposed_ffn_out(B0, node.layer, node.token)
        c_ffn, _ = contributions.get_decomposed_mlp_contributions(
            resid_mid, resid_post, decomposed_ffn
        )

        top_values, top_i = c_ffn.sort(descending=True)
        n = min(self._settings.n_top_neurons, c_ffn.shape[0])
        top_neurons = top_i[0:n].tolist()

        selected_neuron = llm_transparency_tool.components.selector(
            items=[f"{top_neurons[i]}" if i >= 0 else "All" for i in range(-1, n)],
            indices=range(-1, n),
            temperatures=[0.0] + top_values[0:n].tolist(),
            preselected_index=-1,
            key="neuron_selector",
        )
        if selected_neuron is None:
            selected_neuron = -1
        selected_neuron = (
            None if selected_neuron == -1 else top_neurons[selected_neuron]
        )

        return selected_neuron

    @typechecked
    def _draw_token_table(
        self,
        n_top: int,
        n_bottom: int,
        representation: Float[torch.Tensor, "d_model"],
        predecessor: Optional[Float[torch.Tensor, "d_model"]],
    ):
        n_total = n_top + n_bottom

        logits = self._unembed(representation)
        n_vocab = logits.shape[0]
        scores, indices = torch.topk(logits, n_top, largest=True)
        positions = list(range(n_top))

        if n_bottom > 0:
            low_scores, low_indices = torch.topk(logits, n_bottom, largest=False)
            indices = torch.cat((indices, low_indices.flip(0)))
            scores = torch.cat((scores, low_scores.flip(0)))
            positions += range(n_vocab - n_bottom, n_vocab)

        tokens = [string_to_display(w) for w in self._model.tokens_to_strings(indices)]

        if predecessor is not None:
            pre_logits = self._unembed(predecessor)
            _, sorted_pre_indices = pre_logits.sort(descending=True)
            pre_indices_dict = {
                index: pos for pos, index in enumerate(sorted_pre_indices.tolist())
            }
            old_positions = [pre_indices_dict[i] for i in indices.tolist()]

            def pos_gain_string(pos, old_pos):
                if pos == old_pos:
                    return ""
                sign = "↓" if pos > old_pos else "↑"
                return f"({sign}{abs(pos - old_pos)})"

            position_strings = [
                f"{i} {pos_gain_string(i, old_i)}"
                for (i, old_i) in zip(positions, old_positions)
            ]
        else:
            position_strings = [str(pos) for pos in positions]

        def pos_gain_color(s):
            color = "black"
            if isinstance(s, str):
                if "↓" in s:
                    color = "red"
                if "↑" in s:
                    color = "green"
            return f"color: {color}"

        top_df = pd.DataFrame(
            data=zip(position_strings, tokens, scores.tolist()),
            columns=["Pos", "Token", "Score"],
        )

        st.dataframe(
            top_df.style.map(pos_gain_color)
            .background_gradient(
                axis=0,
                cmap=logits_color_map(positive_and_negative=n_bottom > 0),
            )
            .format(precision=3),
            hide_index=True,
            height=self._settings.table_cell_height * (n_total + 1),
            use_container_width=True,
        )

    def draw_token_dynamics(
        self, representation: Float[torch.Tensor, "d_model"], block_name: str
    ) -> None:
        st.write("Promoted/suppressed by")
        st.write(f"{block_name}")
        self._draw_token_table(
            self._settings.n_promoted_tokens,
            self._settings.n_suppressed_tokens,
            representation,
            None,
        )

    def draw_top_tokens(
        self,
        node: UiGraphNode,
        container_top_tokens,
        container_token_dynamics,
    ) -> None:
        pre_node = node.get_residual_predecessor()
        if pre_node is None:
            return

        representation = self._get_representation(node)
        predecessor = self._get_representation(pre_node)

        with container_top_tokens:
            st.write("Top tokens")
            st.write(node.get_name())
            self._draw_token_table(
                self._settings.n_top_tokens,
                0,
                representation,
                predecessor,
            )
        if container_token_dynamics is not None:
            with container_token_dynamics:
                self.draw_token_dynamics(
                    representation - predecessor, node.get_predecessor_block_name()
                )

    def draw_attention_dynamics(self, node: UiGraphNode, head: Optional[int]):
        block_name = node.get_head_name(head)
        block_output = (
            self._model.attention_output_per_head(B0, node.layer, node.token, head)
            if head is not None
            else self._model.attention_output(B0, node.layer, node.token)
        )
        self.draw_token_dynamics(block_output, block_name)

    def draw_ffn_dynamics(self, node: UiGraphNode, neuron: Optional[int]):
        block_name = node.get_neuron_name(neuron)
        block_output = (
            self._model.neuron_output(node.layer, neuron)
            if neuron is not None
            else self._model.ffn_out(node.layer)[B0][node.token]
        )
        self.draw_token_dynamics(block_output, block_name)

    def draw_precision_controls(self, device: str) -> Tuple[torch.dtype, bool]:
        """
        Draw fp16/fp32 switch and AMP control.

        return: The selected precision and whether AMP should be enabled.
        """

        # AMP doesn't work for CPU, so it should be always float32 in CPU.
        if device == GPU:
            precisions = ["float16", "float32"]
            amp_possible = True
        else:
            precisions = ["float32"]
            amp_possible = False

        row_precision = st_row.row([3, 2], vertical_align="bottom")
        dtype = row_precision.selectbox(
            "Precision",
            precisions,
            index=0,
        )
        dtype = torch.float16 if dtype == "float16" else torch.float32
        amp_enabled = row_precision.checkbox(
            "AMP",
            value=amp_possible,
            disabled=not amp_possible,
        )
        return dtype, amp_enabled

    def run_inference(self):
        with st.expander("Model", expanded=True):
            device = st.selectbox(
                "Device",
                possible_devices(),
                index=0,
            )

            dtype, amp_enabled = self.draw_precision_controls(device)

            model_list = sorted(self._config.models.keys())
            try:
                default_choice = model_list.index(self._config.default_model)
            except ValueError:
                st.error(
                    f"Config error: default model {self._config.default_model}"
                    "is not in the list of models"
                )
                return

            model_name = st.selectbox(
                "Model",
                [
                    self._settings.no_model_alt_text if x == "" else x
                    for x in model_list
                ],
                index=default_choice,
            )

            if model_name == self._settings.no_model_alt_text:
                return

            self._model, model_key = load_model_with_session_caching(
                model_name=model_name,
                model_path=self._config.models[model_name],
                device=device,
                dtype=dtype,
            )

            self.draw_model_info()

        with st.expander("Data"):
            sentence = self.draw_dataset_selection()
            if sentence is None:
                return

        with torch.inference_mode():
            with autocast(enabled=amp_enabled, device_type="cuda", dtype=dtype):
                run_model_with_session_caching(self._model, model_key, sentence)

        with st.expander("Graph"):
            self._contribution_threshold = self.draw_threshold_slider()
            self._threshold_mode = self.draw_threshold_mode()
            self._unembedding_mode = self.draw_unembedding_mode()

        with autocast(enabled=amp_enabled, device_type="cuda", dtype=dtype):
            self._graph = get_contribution_graph(
                self._model,
                model_key,
                self._model.tokens()[B0].tolist(),
                (
                    self._contribution_threshold
                    if self._threshold_mode == ThresholdMode.RENORMALIZE
                    else 0.0
                ),
            )

        if self._config.debug:
            with st.expander("System"):
                if device == GPU:
                    self.draw_gpu_memory()

    def draw_graph_and_selection(
        self,
        container_graph,
        container_subblocks,
        container_tokens,
    ) -> None:
        selection = self.draw_graph(
            self._contribution_threshold
            if self._threshold_mode == ThresholdMode.PLAIN
            else 0.0
        )
        if selection is None:
            return
        node = selection.node
        edge = selection.edge

        with container_tokens:
            container_top_tokens, container_token_dynamics = st.columns([1, 1])
        dynamics_already_drawn = False

        if edge is not None and edge.target.type == NodeType.AFTER_ATTN:
            with container_subblocks:
                head = self.draw_attn_info(edge, container_graph)
            with container_token_dynamics:
                self.draw_attention_dynamics(edge.target, head)
                dynamics_already_drawn = True
        elif node is not None and node.type == NodeType.FFN:
            with container_subblocks:
                neuron = self.draw_ffn_info(node)
            with container_token_dynamics:
                self.draw_ffn_dynamics(node, neuron)
                dynamics_already_drawn = True

        if node is not None and node.is_in_residual_stream():
            self.draw_top_tokens(
                node,
                container_top_tokens,
                container_token_dynamics if not dynamics_already_drawn else None,
            )
            dynamics_already_drawn = True

    def run(self) -> None:
        self.prepare_page()
        self.gpu_memory_overhead = init_gpu_memory()

        (
            container_model,
            container_graph,
            container_subblocks,
            container_tokens,
        ) = st.columns(self._settings.column_proportions)

        with container_model:
            self.run_inference()

        if self._model is None or self._graph is None:
            return

        with container_graph:
            self.draw_graph_and_selection(
                container_graph,
                container_subblocks,
                container_tokens,
            )


if __name__ == "__main__":
    top_parser = argparse.ArgumentParser()
    top_parser.add_argument("config_file")
    args = top_parser.parse_args()

    parser = HfArgumentParser([LlmViewerConfig])
    config = parser.parse_json_file(args.config_file)[0]

    app = App(config)
    with Profiler() as prof:
        app.run()

    if config.debug:
        html_code = prof.output_html()
        with st.expander("Profiler"):
            st.components.v1.html(html_code, height=1000)
