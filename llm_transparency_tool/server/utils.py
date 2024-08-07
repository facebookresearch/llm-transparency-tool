# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
    with open(filename) as f:
        dataset = [s.strip("\n") for s in f.readlines()]
    print(f"Loaded {len(dataset)} sentences from {filename}")
    return dataset


@st.cache_resource(
    hash_funcs={
        TransformerLensTransparentLlm: id
    }
)
def load_model(
    model_name: str,
    _device: str,
    _model_path: Optional[str] = None,
    _dtype: torch.dtype = torch.float32,
    supported_model_name: Optional[str] = None,
) -> TransparentLlm:
    """
    Returns the loaded model along with its key. The key is just a unique string which
    can be used later to identify if the model has changed.
    """
    assert _device in possible_devices()

    causal_lm = None
    tokenizer = None

    tl_lm = TransformerLensTransparentLlm(
        model_name=model_name,
        hf_model=causal_lm,
        tokenizer=tokenizer,
        device=_device,
        dtype=_dtype,
        supported_model_name=supported_model_name,
    )

    return tl_lm


def run_model(model: TransparentLlm, sentence: str) -> None:
    print(f"Running inference for '{sentence}'")
    model.run([sentence])


def load_model_with_session_caching(
    **kwargs,
) -> Tuple[TransparentLlm, str]:
    return load_model(**kwargs)

def run_model_with_session_caching(
    _model: TransparentLlm,
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

    run_model(_model, sentence)
    state[LAST_RUN_MODEL_KEY] = model_key
    state[LAST_RUN_SENTENCE] = sentence


@st.cache_resource(
    hash_funcs={
        TransformerLensTransparentLlm: id
    }
)
def get_contribution_graph(
    model: TransparentLlm,  # TODO bug here
    model_key: str,
    tokens: List[str],
    threshold: float,
) -> nx.Graph:
    """
    The `model_key` and `tokens` are used only for caching. The model itself is not
    hashed, hence the `_` in the beginning.
    """
    return llm_transparency_tool.routes.graph.build_full_graph(
        model,
        B0,
        threshold,
    )


def st_placeholder(
    text: str,
    container=st,
    border: bool = True,
    height: Optional[int] = 500,
):
    empty = container.empty()
    empty.container(border=border, height=height).write(f'<small>{text}</small>', unsafe_allow_html=True)
    return empty
