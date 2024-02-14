# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    wget \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user

RUN wget -P /tmp \
    "https://github.com/conda-forge/miniforge/releases/download/23.11.0-0/Mambaforge-23.11.0-0-Linux-x86_64.sh" \
    && bash /tmp/Mambaforge-23.11.0-0-Linux-x86_64.sh -b -p $HOME/mambaforge3 \
    && rm /tmp/Mambaforge-23.11.0-0-Linux-x86_64.sh
ENV PATH $HOME/mambaforge3/bin:$PATH

WORKDIR $HOME

ENV REPO=$HOME/llm-transparency-tool
COPY --chown=user . $REPO

WORKDIR $REPO

RUN mamba env create --name llmtt -f env.yaml -y
ENV PATH $HOME/mambaforge3/envs/llmtt/bin:$PATH
RUN pip install -e .

RUN cd llm_transparency_tool/components/frontend \
    && yarn install \
    && yarn build

EXPOSE 7860
CMD ["streamlit", "run", "llm_transparency_tool/server/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--theme.font=Inconsolata", "--", "config/docker_hosting.json"]
