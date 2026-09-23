"""Opt-in single-split V-JEPA attention backward for eager CUDA diagnostics."""

import torch
import torch.nn.functional as functional
from torch.autograd.function import once_differentiable


class _SingleSplitAttention(torch.autograd.Function):
    @staticmethod
    def forward(context, query, key, value):
        output, logsumexp, seed, offset = torch.ops.aten._scaled_dot_product_efficient_attention(
            query, key, value, None, any(context.needs_input_grad), 0.0, False
        )
        context.save_for_backward(query, key, value, output, logsumexp, seed, offset)
        return output

    @staticmethod
    @once_differentiable
    def backward(context, gradient):
        query, key, value, output, logsumexp, seed, offset = context.saved_tensors
        query_grad, key_grad, value_grad, _ = torch.ops.aten._efficient_attention_backward(
            gradient.transpose(1, 2),
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            None,
            output.transpose(1, 2),
            None,
            None,
            query.shape[2],
            key.shape[2],
            logsumexp,
            0.0,
            seed,
            offset,
            0,
            False,
            num_splits_key=1,
        )
        return query_grad.transpose(1, 2), key_grad.transpose(1, 2), value_grad.transpose(1, 2)


def deterministic_vjepa_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
    """Preserve efficient FP32 forward; pin backward reduction to one split.

    This opt-in path uses private ATen operators from the validated PyTorch
    runtime. It supports unmasked, noncausal FP32 V-JEPA self-attention only;
    unsupported requests fail rather than silently lose the determinism policy.
    CPU tensors use the deterministic math implementation for small-model tests.
    """
    if attn_mask is not None or dropout_p != 0.0 or is_causal:
        raise ValueError("Deterministic V-JEPA SDPA requires no mask, dropout or causal attention")
    if any(tensor.dtype != torch.float32 or tensor.ndim != 4 for tensor in (query, key, value)):
        raise ValueError("Deterministic V-JEPA SDPA requires four-dimensional FP32 tensors")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("Deterministic V-JEPA SDPA requires matching query, key and value shapes")
    if query.device != key.device or query.device != value.device:
        raise ValueError("Deterministic V-JEPA SDPA requires matching devices")
    if query.device.type == "cpu":
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            return functional.scaled_dot_product_attention(query, key, value)
    if query.device.type != "cuda":
        raise ValueError("Deterministic V-JEPA SDPA supports CPU and CUDA only")
    return _SingleSplitAttention.apply(query, key, value)
