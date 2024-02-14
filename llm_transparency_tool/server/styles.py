# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import matplotlib

# Unofficial way do make the padding a bit smaller.
margins_css = """
    <style>
        .main > div {
            padding: 1rem;
            padding-top: 2rem;  # Still need this gap for the top bar
            gap: 0rem;
        }

        section[data-testid="stSidebar"] {
            width: 300px !important; # Set the width to your desired value
        }
    </style>
"""


@dataclass
class RenderSettings:
    column_proportions = [50, 30]

    # We don't know the actual height. This will be used in order to compute the table
    # viewport height when needed.
    table_cell_height = 36

    n_top_tokens = 30
    n_promoted_tokens = 15
    n_suppressed_tokens = 15

    n_top_neurons = 20

    attention_color_map = "Blues"

    no_model_alt_text = "<no model selected>"


def string_to_display(s: str) -> str:
    return s.replace(" ", "·")


def logits_color_map(positive_and_negative: bool) -> matplotlib.colors.Colormap:
    background_colors = {
        "red": [
            [0.0, 0.40, 0.40],
            [0.1, 0.69, 0.69],
            [0.2, 0.83, 0.83],
            [0.3, 0.95, 0.95],
            [0.4, 0.99, 0.99],
            [0.5, 1.0, 1.0],
            [0.6, 0.90, 0.90],
            [0.7, 0.72, 0.72],
            [0.8, 0.49, 0.49],
            [0.9, 0.30, 0.30],
            [1.0, 0.15, 0.15],
        ],
        "green": [
            [0.0, 0.0, 0.0],
            [0.1, 0.09, 0.09],
            [0.2, 0.37, 0.37],
            [0.3, 0.64, 0.64],
            [0.4, 0.85, 0.85],
            [0.5, 1.0, 1.0],
            [0.6, 0.96, 0.96],
            [0.7, 0.88, 0.88],
            [0.8, 0.73, 0.73],
            [0.9, 0.57, 0.57],
            [1.0, 0.39, 0.39],
        ],
        "blue": [
            [0.0, 0.12, 0.12],
            [0.1, 0.16, 0.16],
            [0.2, 0.30, 0.30],
            [0.3, 0.50, 0.50],
            [0.4, 0.78, 0.78],
            [0.5, 1.0, 1.0],
            [0.6, 0.81, 0.81],
            [0.7, 0.52, 0.52],
            [0.8, 0.25, 0.25],
            [0.9, 0.12, 0.12],
            [1.0, 0.09, 0.09],
        ],
    }

    if not positive_and_negative:
        # Stretch the top part to the whole range
        new_colors = {}
        for channel, colors in background_colors.items():
            new_colors[channel] = [
                [(value - 0.5) * 2, color, color]
                for value, color, _ in colors
                if value >= 0.5
            ]
        background_colors = new_colors

    return matplotlib.colors.LinearSegmentedColormap(
        f"RdYG-{positive_and_negative}",
        background_colors,
    )
