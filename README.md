# Transparent LLMs

The goal of the “Transparent LLMs” project is to understand, and therefore demystify, LLMs.

# Installation

```
git clone git@github.com:fairinternal/transparent-llms.git
cd ./transparent-llms

conda create -n transparent_llms_env python=3.10
conda activate transparent_llms_env

pip install -e .
```

# Installing dependencies for contributions

For computing the contributions, graphs etc we need the latest transformer-lens, but
torch should still be 2.0 because of cuda version mismatch. Hence, we install the
dependencies first, and then overwrite the torch version:

```
pip install -r ./requirements_contributions.txt
pip install torch==2.0
```