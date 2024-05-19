import transformers

# If you load a new model from _model_path, please add its class here so that Streamlit can cache it

HASH_FUNCS = {
    transformers.PreTrainedModel: id,
    transformers.PreTrainedTokenizer: id,
    transformers.GPT2LMHeadModel: id,
}
