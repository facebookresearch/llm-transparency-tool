# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import einops
import torch
from jaxtyping import Float
from typeguard import typechecked


@torch.no_grad()
@typechecked
def get_contributions(
    parts: torch.Tensor,
    whole: torch.Tensor,
    distance_norm: int = 1,
) -> torch.Tensor:
    """
    Compute contributions of the `parts` vectors into the `whole` vector.

    Shapes of the tensors are as follows:
    parts:  p_1 ... p_k, v_1 ... v_n, d
    whole:               v_1 ... v_n, d
    result: p_1 ... p_k, v_1 ... v_n

    Here
    * `p_1 ... p_k`: dimensions for enumerating the parts
    * `v_1 ... v_n`: dimensions listing the independent cases (batching),
    * `d` is the dimension to compute the distances on.

    The resulting contributions will be normalized so that
    for each v_: sum(over p_ of result(p_, v_)) = 1.
    """
    EPS = 1e-5

    k = len(parts.shape) - len(whole.shape)
    assert k >= 0
    assert parts.shape[k:] == whole.shape
    bc_whole = whole.expand(parts.shape)  # new dims p_1 ... p_k are added to the front

    distance = torch.nn.functional.pairwise_distance(parts, bc_whole, p=distance_norm)

    whole_norm = torch.norm(whole, p=distance_norm, dim=-1)
    distance = (whole_norm - distance).clip(min=EPS)

    sum = distance.sum(dim=tuple(range(k)), keepdim=True)

    return distance / sum


@torch.no_grad()
@typechecked
def get_contributions_with_one_off_part(
    parts: torch.Tensor,
    one_off: torch.Tensor,
    whole: torch.Tensor,
    distance_norm: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Same as computing the contributions, but there is one additional part. That's useful
    because we always have the residual stream as one of the parts.

    See `get_contributions` documentation about `parts` and `whole` dimensions. The
    `one_off` should have the same dimensions as `whole`.

    Returns a pair consisting of
    1. contributions tensor for the `parts`
    2. contributions tensor for the `one_off` vector
    """
    assert one_off.shape == whole.shape

    k = len(parts.shape) - len(whole.shape)
    assert k >= 0

    # Flatten the p_ dimensions, get contributions for the list, unflatten.
    flat = parts.flatten(start_dim=0, end_dim=k - 1)
    flat = torch.cat([flat, one_off.unsqueeze(0)])
    contributions = get_contributions(flat, whole, distance_norm)
    parts_contributions, one_off_contributions = torch.split(
        contributions, flat.shape[0] - 1
    )
    return (
        parts_contributions.unflatten(0, parts.shape[0:k]),
        one_off_contributions[0],
    )


@torch.no_grad()
@typechecked
def get_attention_contributions(
    resid_pre: Float[torch.Tensor, "batch pos d_model"],
    resid_mid: Float[torch.Tensor, "batch pos d_model"],
    decomposed_attn: Float[torch.Tensor, "batch pos key_pos head d_model"],
    distance_norm: int = 1,
) -> Tuple[
    Float[torch.Tensor, "batch pos key_pos head"],
    Float[torch.Tensor, "batch pos"],
]:
    """
    Returns a pair of
    - a tensor of contributions of each token via each head
    - the contribution of the residual stream.
    """

    # part dimensions | batch dimensions | vector dimension
    # ----------------+------------------+-----------------
    # key_pos, head   | batch, pos       | d_model
    parts = einops.rearrange(
        decomposed_attn,
        "batch pos key_pos head d_model -> key_pos head batch pos d_model",
    )
    attn_contribution, residual_contribution = get_contributions_with_one_off_part(
        parts, resid_pre, resid_mid, distance_norm
    )
    return (
        einops.rearrange(
            attn_contribution, "key_pos head batch pos -> batch pos key_pos head"
        ),
        residual_contribution,
    )


@torch.no_grad()
@typechecked
def get_mlp_contributions(
    resid_mid: Float[torch.Tensor, "batch pos d_model"],
    resid_post: Float[torch.Tensor, "batch pos d_model"],
    mlp_out: Float[torch.Tensor, "batch pos d_model"],
    distance_norm: int = 1,
) -> Tuple[Float[torch.Tensor, "batch pos"], Float[torch.Tensor, "batch pos"]]:
    """
    Returns a pair of (mlp, residual) contributions for each sentence and token.
    """

    contributions = get_contributions(
        torch.stack((mlp_out, resid_mid)), resid_post, distance_norm
    )
    return contributions[0], contributions[1]


@torch.no_grad()
@typechecked
def get_decomposed_mlp_contributions(
    resid_mid: Float[torch.Tensor, "d_model"],
    resid_post: Float[torch.Tensor, "d_model"],
    decomposed_mlp_out: Float[torch.Tensor, "hidden d_model"],
    distance_norm: int = 1,
) -> Tuple[Float[torch.Tensor, "hidden"], float]:
    """
    Similar to `get_mlp_contributions`, but it takes the MLP output for each neuron of
    the hidden layer and thus computes a contribution per neuron.

    Doesn't contain batch and token dimensions for sake of saving memory. But we may
    consider adding them.
    """

    neuron_contributions, residual_contribution = get_contributions_with_one_off_part(
        decomposed_mlp_out, resid_mid, resid_post, distance_norm
    )
    return neuron_contributions, residual_contribution.item()


@torch.no_grad()
def apply_threshold_and_renormalize(
    threshold: float,
    c_blocks: torch.Tensor,
    c_residual: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Thresholding mechanism used in the original graphs paper. After the threshold is
    applied, the remaining contributions are renormalized on order to sum up to 1 for
    each representation.

    threshold: The threshold.
    c_residual: Contribution of the residual stream for each representation. This tensor
        should contain 1 element per representation, i.e., its dimensions are all batch
        dimensions.
    c_blocks: Contributions of the blocks. Could be 1 block per representation, like
        ffn, or heads*tokens blocks in case of attention. The shape of `c_residual`
        must be a prefix if the shape of this tensor. The remaining dimensions are for
        listing the blocks.
    """

    block_dims = len(c_blocks.shape)
    resid_dims = len(c_residual.shape)
    bound_dims = block_dims - resid_dims
    assert bound_dims >= 0
    assert c_blocks.shape[0:resid_dims] == c_residual.shape

    c_blocks = c_blocks * (c_blocks > threshold)
    c_residual = c_residual * (c_residual > threshold)

    denom = c_residual + c_blocks.sum(dim=tuple(range(resid_dims, block_dims)))
    return (
        c_blocks / denom.reshape(denom.shape + (1,) * bound_dims),
        c_residual / denom,
    )
