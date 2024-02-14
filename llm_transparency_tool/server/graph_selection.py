# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Dict, Optional

from llm_transparency_tool.routes.graph_node import GraphNode, NodeType


class UiGraphNode(GraphNode):
    @staticmethod
    def from_json(json: Dict[str, Any]) -> Optional["UiGraphNode"]:
        try:
            layer = json["cell"]["layer"]
            token = json["cell"]["token"]
            type = NodeType(json["item"])
            return UiGraphNode(layer, token, type)
        except (TypeError, KeyError):
            return None


@dataclass
class UiGraphEdge:
    source: UiGraphNode
    target: UiGraphNode
    weight: float

    @staticmethod
    def from_json(json: Dict[str, Any]) -> Optional["UiGraphEdge"]:
        try:
            source = UiGraphNode.from_json(json["from"])
            target = UiGraphNode.from_json(json["to"])
            if source is None or target is None:
                return None
            weight = float(json["weight"])
            return UiGraphEdge(source, target, weight)
        except (TypeError, KeyError):
            return None


@dataclass
class GraphSelection:
    node: Optional[UiGraphNode]
    edge: Optional[UiGraphEdge]

    @staticmethod
    def from_json(json: Dict[str, Any]) -> Optional["GraphSelection"]:
        try:
            node = UiGraphNode.from_json(json["node"])
            edge = UiGraphEdge.from_json(json["edge"])
            return GraphSelection(node, edge)
        except (TypeError, KeyError):
            return None
