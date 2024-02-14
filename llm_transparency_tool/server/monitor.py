# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import streamlit as st
from pyinstrument import Profiler
from typing import Dict
import pandas as pd


@st.cache_resource(max_entries=1, show_spinner=False)
def init_gpu_memory():
    """
    When CUDA is initialized, it occupies some memory on the GPU thus this overhead
    can sometimes make it difficult to understand how much memory is actually used by
    the model.

    This function is used to initialize CUDA and measure the overhead.
    """
    if not torch.cuda.is_available():
        return {}

    # lets init torch gpu for a moment
    gpu_memory_overhead = {}
    for i in range(torch.cuda.device_count()):
        torch.ones(1).cuda(i)
        free, total = torch.cuda.mem_get_info(i)
        occupied = total - free
        gpu_memory_overhead[i] = occupied

    return gpu_memory_overhead


class SystemMonitor:
    """
    This class is used to monitor the system resources such as GPU memory and CPU
    usage. It uses the pyinstrument library to profile the code and measure the
    execution time of different parts of the code.
    """

    def __init__(
        self,
        enabled: bool = False,
    ):
        self.enabled = enabled
        self.profiler = Profiler()
        self.overhead: Dict[int, int]

    def __enter__(self):
        if not self.enabled:
            return

        self.overhead = init_gpu_memory()

        self.profiler.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.enabled:
            return

        self.profiler.__exit__(exc_type, exc_value, traceback)

        self.report_gpu_usage()
        self.report_profiler()

        with st.expander("Session state"):
            st.write(st.session_state)

        return None

    def report_gpu_usage(self):

        if not torch.cuda.is_available():
            return

        data = []

        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            occupied = total - free
            data.append({
                'overhead': self.overhead[i],
                'occupied': occupied - self.overhead[i],
                'free': free,
            })
        df = pd.DataFrame(data, columns=["overhead", "occupied", "free"])

        with st.sidebar.expander("System"):
            st.write("GPU memory on server")
            df /= 1024 ** 3  # Convert to GB
            st.bar_chart(df, width=200, height=200, color=["#fefefe", "#84c9ff", "#fe2b2b"])

    def report_profiler(self):
        html_code = self.profiler.output_html()
        with st.expander("Profiler", expanded=False):
            st.components.v1.html(html_code, height=1000, scrolling=True)
