<h1>
  <img width="500" alt="LLM Transparency Tool" src="https://github.com/facebookresearch/llm-transparency-tool/assets/1367529/795233be-5ef7-4523-8282-67486cf2e15f">
</h1>

<img width="832" alt="screenshot" src="https://github.com/facebookresearch/llm-transparency-tool/assets/1367529/78f6f9e2-fe76-4ded-bb78-a57f64f4ac3a">


## Key functionality

* Choose your model, choose or add your prompt, run the inference.
* Browse contribution graph.
    * Select the token to build the graph from.
    * Tune the contribution threshold.
* Select representation of any token after any block.
* For the representation, see its projection to the output vocabulary, see which tokens
were promoted/suppressed but the previous block.
* The following things are clickable:
  * Edges. That shows more info about the contributing attention head.
  * Heads when an edge is selected. You can see what this head is promoting/suppressing.
  * FFN blocks (little squares on the graph).
  * Neurons when an FFN block is selected.


## Installation

### Dockerized running
```bash
# From the repository root directory
docker build -t llm_transparency_tool .
docker run --rm -p 7860:7860 llm_transparency_tool
```

### Local Installation


```bash
# download
git clone git@github.com:facebookresearch/llm-transparency-tool.git
cd llm-transparency-tool

# install the necessary packages
conda env create --name llmtt -f env.yaml
# install the `llm_transparency_tool` package
pip install -e .

# now, we need to build the frontend
# don't worry, even `yarn` comes preinstalled by `env.yaml`
cd llm_transparency_tool/components/frontend
yarn install
yarn build
```

### Launch

```bash
streamlit run llm_transparency_tool/server/app.py -- config/local.json
```


## Adding support for your LLM

Initially, the tool allows you to select from just a handful of models. Here are the
options you can try for using your model in the tool, from least to most
effort.


### The model is already supported by TransformerLens

Full list of models is [here](https://github.com/neelnanda-io/TransformerLens/blob/0825c5eb4196e7ad72d28bcf4e615306b3897490/transformer_lens/loading_from_pretrained.py#L18).
In this case, the model can be added to the configuration json file.


### Tuned version of a model supported by TransformerLens

Add the official name of the model to the config along with the location to read the
weights from.


### The model is not supported by TransformerLens

In this case the UI wouldn't know how to create proper hooks for the model. You'd need
to implement your version of [TransparentLlm](./llm_transparency_tool/models/transparent_llm.py#L28) class and alter the
Streamlit app to use your implementation.

## Citation
If you use the LLM Transparency Tool for your research, please consider citing:

```bibtex
@article{tufanov2024lm,
      title={LM Transparency Tool: Interactive Tool for Analyzing Transformer Language Models}, 
      author={Igor Tufanov and Karen Hambardzumyan and Javier Ferrando and Elena Voita},
      year={2024},
      journal={Arxiv},
      url={https://arxiv.org/abs/2404.07004}
}

@article{ferrando2024information,
    title={Information Flow Routes: Automatically Interpreting Language Models at Scale}, 
    author={Javier Ferrando and Elena Voita},
    year={2024},
    journal={Arxiv},
    url={https://arxiv.org/abs/2403.00824}
}
````

## License

This code is made available under a [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license, as found in the LICENSE file.
However you may have other legal obligations that govern your use of other content, such as the terms of service for third-party models.
