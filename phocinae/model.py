"""Pure-torch mmBERT-small encoder + typed-decisions head.

Re-implements (inference only, no runtime dependency on laya/transformers):
  - mmBERT-small encoder: 384 hidden / 22 layers / 6 heads / GEGLU MLP /
    RoPE (Gemma2-style, theta=160000) / sliding-window attention on all but
    every third layer (full attention) / pre-norm, LayerNorm(eps=1e-5, no bias),
    layer 0 attention norm is Identity.
  - decision head: type embedding + 2-layer pre-norm TransformerEncoder
    (nn.TransformerEncoderLayer, d=384, 6 heads, ff=1536, dropout 0.1) +
    marker-gathered scorer + act head (d+4 -> 256 -> 2).

State-dict keys match the released checkpoint exactly
(encoder.layers.i.attn.Wqkv/Wo, mlp.Wi/Wo, attn_norm/mlp_norm, head.layers.*,
scorer.*, act_head.*, type_emb.weight).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)


class _Attn(nn.Module):
    """Fused Wqkv attention with RoPE + SDPA (checkpoint keys attn.Wqkv/attn.Wo)."""

    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.hd = d // heads
        self.Wqkv = nn.Linear(d, 3 * d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)

    def forward(self, x, mask, cos, sin):
        B, L, _ = x.shape
        qkv = self.Wqkv(x).view(B, L, 3, self.heads, self.hd)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = (q.float() * cos + _rotate_half(q.float()) * sin).to(q.dtype)
        k = (k.float() * cos + _rotate_half(k.float()) * sin).to(k.dtype)
        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False,
            scale=1.0 / math.sqrt(self.hd))
        return self.Wo(o.transpose(1, 2).reshape(B, L, -1))


class _MLP(nn.Module):
    """GEGLU MLP (checkpoint keys mlp.Wi/mlp.Wo): Wo(gelu(input) * gate)."""

    def __init__(self, d, inter):
        super().__init__()
        self.Wi = nn.Linear(d, 2 * inter, bias=False)
        self.Wo = nn.Linear(inter, d, bias=False)

    def forward(self, x):
        inp, gate = self.Wi(x).chunk(2, dim=-1)
        return self.Wo(F.gelu(inp) * gate)


class _EncoderLayer(nn.Module):
    def __init__(self, d, heads, inter, eps, full, first):
        super().__init__()
        self.full = full
        self.attn_norm = nn.Identity() if first else nn.LayerNorm(d, eps=eps, bias=False)
        self.attn = _Attn(d, heads)
        self.mlp_norm = nn.LayerNorm(d, eps=eps, bias=False)
        self.mlp = _MLP(d, inter)

    def forward(self, x, mask, cos, sin):
        x = x + self.attn(self.attn_norm(x), mask, cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class MMBertEncoder(nn.Module):
    def __init__(self, vocab, d, n_layers, heads, inter, eps, full_layers,
                 theta, local_window):
        super().__init__()
        self.hd = d // heads
        self.local_window = local_window
        self.embeddings = nn.Module()
        self.embeddings.tok_embeddings = nn.Embedding(vocab, d, padding_idx=0)
        self.embeddings.norm = nn.LayerNorm(d, eps=eps, bias=False)
        self.layers = nn.ModuleList([
            _EncoderLayer(d, heads, inter, eps, full=(i in full_layers), first=(i == 0))
            for i in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d, eps=eps, bias=False)
        inv = 1.0 / (theta ** (torch.arange(0, self.hd, 2, dtype=torch.float32) / self.hd))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, input_ids, attention_mask):
        B, L = input_ids.shape
        dev = input_ids.device
        x = self.embeddings.tok_embeddings(input_ids)
        x = self.embeddings.norm(x)
        pos = torch.arange(L, device=dev, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq.to(dev))  # [L, hd/2] fp32
        emb = torch.cat([freqs, freqs], dim=-1)          # [L, hd]
        cos = emb.cos()[None, None]                      # [1, 1, L, hd]
        sin = emb.sin()[None, None]
        pad = attention_mask.bool()                      # [B, L]
        full_mask = pad[:, None, None, :]                # [B, 1, 1, L]
        band = (pos[:, None] - pos[None, :]).abs() <= self.local_window
        slide_mask = band[None, None] & pad[:, None, None, :]  # [B, 1, L, L]
        for layer in self.layers:
            x = layer(x, full_mask if layer.full else slide_mask, cos, sin)
        return self.final_norm(x)


class DecisionModel(nn.Module):
    """Full decision model; checkpoint keys live at the top level:

    encoder.*, head.layers.*, scorer.*, act_head.*, type_emb.weight
    """

    def __init__(self, d, head_layers, heads, dropout):
        super().__init__()
        self.type_emb = nn.Embedding(3, d)
        layer = nn.TransformerEncoderLayer(
            d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(
            layer, head_layers, enable_nested_tensor=False)
        self.scorer = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(
            nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, 2))

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        h = self.encoder(input_ids, attention_mask)
        h = h + self.type_emb(qtype)[:, None, :]
        pad = ~attention_mask.bool()
        h = self.head(h, src_key_padding_mask=pad)
        return self._score(h, marker_pos, marker_mask)

    def _score(self, h, marker_pos, marker_mask):
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)
        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        if p.size(-1) >= 2:
            top2 = p.topk(2, -1).values
        else:
            top1 = p.topk(1, -1).values
            top2 = torch.cat([top1, torch.zeros_like(top1)], dim=-1)
        feats = torch.stack(
            [top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        act_logits = self.act_head(torch.cat([h[:, 0].float(), feats], -1))
        return logits, act_logits
