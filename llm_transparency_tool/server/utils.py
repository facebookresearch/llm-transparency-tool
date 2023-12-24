import uuid
from typing import List, Optional, Tuple

import networkx as nx
import streamlit as st
import torch
import transformers

import llm_transparency_tool.routes.graph
from llm_transparency_tool.models.tlens_model import TransformerLensTransparentLlm
from llm_transparency_tool.models.transparent_llm import TransparentLlm

GPU = "gpu"
CPU = "cpu"

# This variable is for expressing the idea that batch_id = 0, but make it more
# readable than just 0.
B0 = 0


def possible_devices() -> List[str]:
    devices = []
    if torch.cuda.is_available():
        devices.append("gpu")
    devices.append("cpu")
    return devices


def load_dataset(filename) -> List[str]:
    with open(filename, "r+t") as f:
        dataset = [s.strip("\n") for s in f.readlines()]
    print(f"Loaded {len(dataset)} sentences from {filename}")
    return dataset


@st.cache_resource(max_entries=1, show_spinner=True)
def init_gpu_memory():
    # lets init torch gpu for a moment
    gpu_memory_overhead = {}
    for i in range(torch.cuda.device_count()):
        torch.ones(1).cuda(i)
        [free, total] = [x / 1024 / 1024 for x in torch.cuda.mem_get_info(i)]
        occupied = total - free
        gpu_memory_overhead[i] = occupied
    return gpu_memory_overhead


def load_model(
    model_name: str,
    device: str,
    model_path: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[TransparentLlm, str]:
    """
    Returns the loaded model along with its key. The key is just a unique string which
    can be used later to identify if the model has changed.
    """
    assert device in possible_devices()

    causal_lm = None
    tokenizer = None

    # TODO(igortufanov): Figure out what's different for Llama.
    if model_name.startswith("meta-llama/Llama-2-") or model_path is not None:
        device_map = "auto" if device == GPU else "cpu"
        causal_lm = transformers.AutoModelForCausalLM.from_pretrained(
            model_path or model_name,
            device_map=device_map,
            torch_dtype=dtype,
        )

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path or model_name)
        tokenizer.padding_side = "left"
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    tl_lm = TransformerLensTransparentLlm(
        model_name=model_name,
        hf_model=causal_lm,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
    )

    return tl_lm, str(uuid.uuid4())


def run_model(model: TransparentLlm, sentence: str) -> None:
    print(f"Running inference for '{sentence}'")
    model.run([sentence])


def load_model_with_session_caching(
    **kwargs,
) -> Tuple[TransparentLlm, str]:
    """
    Loads model with simple version of caching. Each user's model is stored in
    streamlit session state.

    We're not using cache_resource so far because:
    1. the model class is not thread-safe
    2. each user has a different intermediate state tensors

    Args:
    - kwargs should be the aguments of the `load_model` function.
    """
    LAST_MODEL_ARGS = "last_model_args"
    LAST_MODEL = "last_model"
    LAST_MODEL_KEY = "last_model_key"
    state = st.session_state

    if state.get(LAST_MODEL_ARGS, None) == kwargs:
        return state[LAST_MODEL], state[LAST_MODEL_KEY]

    model, key = load_model(**kwargs)
    state[LAST_MODEL] = model
    state[LAST_MODEL_KEY] = key
    state[LAST_MODEL_ARGS] = kwargs
    return model, key


def run_model_with_session_caching(
    model: TransparentLlm,
    model_key: str,
    sentence: str,
):
    LAST_RUN_MODEL_KEY = "last_run_model_key"
    LAST_RUN_SENTENCE = "last_run_sentence"
    state = st.session_state

    if (
        state.get(LAST_RUN_MODEL_KEY, None) == model_key
        and state.get(LAST_RUN_SENTENCE, None) == sentence
    ):
        return

    run_model(model, sentence)
    state[LAST_RUN_MODEL_KEY] = model_key
    state[LAST_RUN_SENTENCE] = sentence


@st.cache_data(max_entries=100)
def get_contribution_graph(
    _model: TransparentLlm,
    model_key: str,
    tokens: List[str],
    threshold: float,
) -> List[nx.Graph]:
    """
    The `model_key` and `tokens` are used only for caching. The model itself is not
    hashed, hence the `_` in the beginning.
    """
    return llm_transparency_tool.routes.graph.build_full_graph(
        _model,
        B0,
        threshold,
    )
