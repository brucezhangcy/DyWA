"""Pure-PyTorch drop-in for flash_attn v1's `flash_attn.flash_attention` module.

WHY THIS EXISTS
    The DyWA checkpoints were trained inside a Docker image (pkm1:v0) that had the
    CUDA `flash-attn` (v1) package. The bare-metal conda env `dywa_render` used for
    eval on coldbrew does not, and the user has no docker access. Rather than modify
    repo code (which is off-limits), this shim satisfies
        `from flash_attn.flash_attention import FlashAttention, FlashMHA`
    from a directory placed first on PYTHONPATH.

CORRECTNESS
    FlashAttention is an *exact* (memory-efficient) implementation of standard scaled
    dot-product softmax attention. This shim computes the identical math in plain
    PyTorch, so results are numerically equivalent up to floating-point rounding — it
    is a faithful substitute, not an approximation. Class/submodule names (`Wqkv`,
    `inner_attn`, `out_proj`) match flash_attn v1 exactly so checkpoint state_dicts
    load without remapping. See dywa_research/progress_log_size_transition.md.

Interfaces replicated (flash_attn v1.x):
  * FlashAttention(softmax_scale=None, attention_dropout=0.0)
        .forward(qkv, key_padding_mask=None, causal=False, need_weights=False, ...)
        qkv: (B, S, 3, H, D) -> returns (context: (B, S, H, D), attn_weights|None)
  * FlashMHA(embed_dim, num_heads, bias=True, attention_dropout=0.0, causal=False, ...)
        submodules Wqkv / inner_attn / out_proj
        .forward(x, key_padding_mask=None, need_weights=False) -> (out, attn|None)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _sdpa(q, k, v, softmax_scale=None, causal=False,
          key_padding_mask=None, dropout_p=0.0, training=False,
          need_weights=False):
    """q,k,v: (B, S, H, D). Returns (out (B,S,H,D), attn|None)."""
    B, Sq, H, D = q.shape
    Sk = k.shape[1]
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(D)
    q = q.permute(0, 2, 1, 3)  # (B,H,Sq,D)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    dots = torch.matmul(q, k.transpose(-1, -2)) * scale  # (B,H,Sq,Sk)
    neg = torch.finfo(dots.dtype).min
    if causal:
        cm = torch.triu(torch.ones(Sq, Sk, dtype=torch.bool, device=q.device),
                        diagonal=1)
        dots = dots.masked_fill(cm[None, None], neg)
    if key_padding_mask is not None:
        # flash_attn convention: True/1 = keep (valid); mask out the rest.
        invalid = ~key_padding_mask.to(torch.bool)  # (B, Sk)
        dots = dots.masked_fill(invalid[:, None, None, :], neg)
    attn = torch.softmax(dots, dim=-1)
    if dropout_p > 0.0 and training:
        attn = F.dropout(attn, p=dropout_p)
    out = torch.matmul(attn, v)  # (B,H,Sq,D)
    out = out.permute(0, 2, 1, 3).contiguous()  # (B,Sq,H,D)
    return out, (attn if need_weights else None)


class FlashAttention(nn.Module):
    def __init__(self, softmax_scale=None, attention_dropout=0.0, **kwargs):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, qkv, key_padding_mask=None, causal=False,
                cu_seqlens=None, max_s=None, need_weights=False, **kwargs):
        # qkv: (B, S, 3, H, D)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        return _sdpa(q, k, v, softmax_scale=self.softmax_scale, causal=causal,
                     key_padding_mask=key_padding_mask, dropout_p=self.dropout_p,
                     training=self.training, need_weights=need_weights)


class FlashMHA(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, bias=True, batch_first=True,
                 attention_dropout=0.0, causal=False, softmax_scale=None,
                 device=None, dtype=None, **kwargs):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        factory = {}
        if device is not None:
            factory['device'] = device
        if dtype is not None:
            factory['dtype'] = dtype
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory)
        self.inner_attn = FlashAttention(softmax_scale=softmax_scale,
                                         attention_dropout=attention_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory)

    def forward(self, x, key_padding_mask=None, need_weights=False, **kwargs):
        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d',
                        three=3, h=self.num_heads)
        context, attn = self.inner_attn(qkv, key_padding_mask=key_padding_mask,
                                        need_weights=need_weights,
                                        causal=self.causal)
        out = self.out_proj(rearrange(context, 'b s h d -> b s (h d)'))
        return out, attn


# Some code paths import FlashMHA for a full MHA; expose common aliases too.
__all__ = ['FlashAttention', 'FlashMHA']
