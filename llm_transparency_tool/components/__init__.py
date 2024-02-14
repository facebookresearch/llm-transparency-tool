# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import List, Optional

import networkx as nx
import streamlit.components.v1 as components

from llm_transparency_tool.models.transparent_llm import ModelInfo
from llm_transparency_tool.server.graph_selection import GraphSelection, UiGraphNode

_RELEASE = True

if _RELEASE:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    config = {
        "path": os.path.join(parent_dir, "frontend/build"),
    }
else:
    config = {
        "url": "http://localhost:3001",
    }

_component_func = components.declare_component("contribution_graph", **config)


def is_node_valid(node: UiGraphNode, n_layers: int, n_tokens: int):
    return node.layer < n_layers and node.token < n_tokens


def is_selection_valid(s: GraphSelection, n_layers: int, n_tokens: int):
    if not s:
        return True
    if s.node:
        if not is_node_valid(s.node, n_layers, n_tokens):
            return False
    if s.edge:
        for node in [s.edge.source, s.edge.target]:
            if not is_node_valid(node, n_layers, n_tokens):
                return False
    return True


def contribution_graph(
    model_info: ModelInfo,
    tokens: List[str],
    graphs: List[nx.Graph],
    key: str,
) -> Optional[GraphSelection]:
    """Create a new instance of contribution graph.

    Returns selected graph node or None if nothing was selected.
    """
    assert len(tokens) == len(graphs)

    result = _component_func(
        component="graph",
        model_info=model_info.__dict__,
        tokens=tokens,
        edges_per_token=[nx.node_link_data(g)["links"] for g in graphs],
        default=None,
        key=key,
    )

    selection = GraphSelection.from_json(result)

    n_tokens = len(tokens)
    n_layers = model_info.n_layers
    # We need this extra protection because even though the component has to check for
    # the validity of the selection, sometimes it allows invalid output. It's some
    # unexpected effect that has something to do with React and how the output value is
    # set for the component.
    if not is_selection_valid(selection, n_layers, n_tokens):
        selection = None

    return selection


def selector(
    items: List[str],
    indices: List[int],
    temperatures: Optional[List[float]],
    preselected_index: Optional[int],
    key: str,
) -> Optional[int]:
    """Create a new instance of selector.

    Returns selected item index.
    """
    n = len(items)
    assert n == len(indices)
    items = [{"index": i, "text": s} for s, i in zip(items, indices)]

    if temperatures is not None:
        assert n == len(temperatures)
        for i, t in enumerate(temperatures):
            items[i]["temperature"] = t

    result = _component_func(
        component="selector",
        items=items,
        preselected_index=preselected_index,
        default=None,
        key=key,
    )

    return None if result is None else int(result)
