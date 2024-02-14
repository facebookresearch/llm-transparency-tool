# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import networkx as nx
import torch

import llm_transparency_tool.routes.contributions as contributions
from llm_transparency_tool.models.transparent_llm import TransparentLlm


class GraphBuilder:
    """
    Constructs the contributions graph with edges given one by one. The resulting graph
    is a networkx graph that can be accessed via the `graph` field. It contains the
    following types of nodes:

    - X0_<token>: the original token.
    - A<layer>_<token>: the residual stream after attention at the given layer for the
        given token.
    - M<layer>_<token>: the ffn block.
    - I<layer>_<token>: the residual stream after the ffn block.
    """

    def __init__(self, n_layers: int, n_tokens: int):
        self._n_layers = n_layers
        self._n_tokens = n_tokens

        self.graph = nx.DiGraph()
        for layer in range(n_layers):
            for token in range(n_tokens):
                self.graph.add_node(f"A{layer}_{token}")
                self.graph.add_node(f"I{layer}_{token}")
                self.graph.add_node(f"M{layer}_{token}")
        for token in range(n_tokens):
            self.graph.add_node(f"X0_{token}")

    def get_output_node(self, token: int):
        return f"I{self._n_layers - 1}_{token}"

    def _add_edge(self, u: str, v: str, weight: float):
        # TODO(igortufanov): Here we sum up weights for multi-edges. It happens with
        # attention from the current token and the residual edge. Ideally these need to
        # be 2 separate edges, but then we need to do a MultiGraph. Multigraph is fine,
        # but when we try to traverse it, we face some NetworkX issue with EDGE_OK
        # receiving 3 arguments instead of 2.
        if self.graph.has_edge(u, v):
            self.graph[u][v]["weight"] += weight
        else:
            self.graph.add_edge(u, v, weight=weight)

    def add_attention_edge(self, layer: int, token_from: int, token_to: int, w: float):
        self._add_edge(
            f"I{layer-1}_{token_from}" if layer > 0 else f"X0_{token_from}",
            f"A{layer}_{token_to}",
            w,
        )

    def add_residual_to_attn(self, layer: int, token: int, w: float):
        self._add_edge(
            f"I{layer-1}_{token}" if layer > 0 else f"X0_{token}",
            f"A{layer}_{token}",
            w,
        )

    def add_ffn_edge(self, layer: int, token: int, w: float):
        self._add_edge(f"A{layer}_{token}", f"M{layer}_{token}", w)
        self._add_edge(f"M{layer}_{token}", f"I{layer}_{token}", w)

    def add_residual_to_ffn(self, layer: int, token: int, w: float):
        self._add_edge(f"A{layer}_{token}", f"I{layer}_{token}", w)


@torch.no_grad()
def build_full_graph(
    model: TransparentLlm,
    batch_i: int = 0,
    renormalizing_threshold: Optional[float] = None,
) -> nx.Graph:
    """
    Build the contribution graph for all blocks of the model and all tokens.

    model: The transparent llm which already did the inference.
    batch_i: Which sentence to use from the batch that was given to the model.
    renormalizing_threshold: If specified, will apply renormalizing thresholding to the
    contributions. All contributions below the threshold will be erazed and the rest
    will be renormalized.
    """
    n_layers = model.model_info().n_layers
    n_tokens = model.tokens()[batch_i].shape[0]

    builder = GraphBuilder(n_layers, n_tokens)

    for layer in range(n_layers):
        c_attn, c_resid_attn = contributions.get_attention_contributions(
            resid_pre=model.residual_in(layer)[batch_i].unsqueeze(0),
            resid_mid=model.residual_after_attn(layer)[batch_i].unsqueeze(0),
            decomposed_attn=model.decomposed_attn(batch_i, layer).unsqueeze(0),
        )
        if renormalizing_threshold is not None:
            c_attn, c_resid_attn = contributions.apply_threshold_and_renormalize(
                renormalizing_threshold, c_attn, c_resid_attn
            )
        for token_from in range(n_tokens):
            for token_to in range(n_tokens):
                # Sum attention contributions over heads.
                c = c_attn[batch_i, token_to, token_from].sum().item()
                builder.add_attention_edge(layer, token_from, token_to, c)
        for token in range(n_tokens):
            builder.add_residual_to_attn(
                layer, token, c_resid_attn[batch_i, token].item()
            )

        c_ffn, c_resid_ffn = contributions.get_mlp_contributions(
            resid_mid=model.residual_after_attn(layer)[batch_i].unsqueeze(0),
            resid_post=model.residual_out(layer)[batch_i].unsqueeze(0),
            mlp_out=model.ffn_out(layer)[batch_i].unsqueeze(0),
        )
        if renormalizing_threshold is not None:
            c_ffn, c_resid_ffn = contributions.apply_threshold_and_renormalize(
                renormalizing_threshold, c_ffn, c_resid_ffn
            )
        for token in range(n_tokens):
            builder.add_ffn_edge(layer, token, c_ffn[batch_i, token].item())
            builder.add_residual_to_ffn(
                layer, token, c_resid_ffn[batch_i, token].item()
            )

    return builder.graph


def build_paths_to_predictions(
    graph: nx.Graph,
    n_layers: int,
    n_tokens: int,
    starting_tokens: List[int],
    threshold: float,
) -> List[nx.Graph]:
    """
    Given the full graph, this function returns only the trees leading to the specified
    tokens. Edges with weight below `threshold` will be ignored.
    """
    builder = GraphBuilder(n_layers, n_tokens)

    rgraph = graph.reverse()
    search_graph = nx.subgraph_view(
        rgraph, filter_edge=lambda u, v: rgraph[u][v]["weight"] > threshold
    )

    result = []
    for start in starting_tokens:
        assert start < n_tokens
        assert start >= 0
        edges = nx.edge_dfs(search_graph, source=builder.get_output_node(start))
        tree = search_graph.edge_subgraph(edges)
        # Reverse the edges because the dfs was going from upper layer downwards.
        result.append(tree.reverse())

    return result
