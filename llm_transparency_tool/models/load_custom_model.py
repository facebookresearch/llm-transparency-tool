import torch
import transformers
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.loading_from_pretrained import convert_gpt2_weights


def hf_model_to_hooked_transformer(
    hf_model: transformers.PreTrainedModel,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> HookedTransformer:
    """
    Returns a converted HookedTransformer from a HuggingFace model. 
    Currently, only GPT2LMHeadModel is supported. 
    However, you can add support for other models by using the following implementation as a reference.
    """
    architecture = hf_model.config.architectures[0]
    hf_config = hf_model.config
    if architecture == "GPT2LMHeadModel":
        hooked_cfg = HookedTransformerConfig(
            d_model=hf_config.n_embd,
            d_head=hf_config.n_embd // hf_config.n_head,
            d_mlp=hf_config.n_embd * 4,
            n_layers=hf_config.n_layer,
            n_ctx=hf_config.n_ctx,
            eps=hf_config.layer_norm_epsilon,
            d_vocab=hf_config.vocab_size,
            act_fn=hf_config.activation_function,
            use_attn_scale=True,
            use_local_attn=False,
            scale_attn_by_inverse_layer_idx=hf_config.scale_attn_by_inverse_layer_idx,
            normalization_type="LN",
            device=device,
            dtype=dtype,
        )
        hooked_weights = convert_gpt2_weights(hf_model, hooked_cfg)
    else:
        raise NotImplementedError(
            f"Loading custom {architecture} is not currently supported"
        )

    hooked_transformer = HookedTransformer(hooked_cfg)
    hooked_transformer.load_and_process_state_dict(
        hooked_weights,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    return hooked_transformer
