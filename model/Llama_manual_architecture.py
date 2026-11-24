# Llama_manual_architecture.py
import torch.nn as nn
import torch
import math
import importlib
import srs.LLM.config as cfg_module
import torch.nn.functional as F

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        emb, hid = cfg["emb_dim"], cfg["hidden_dim"]
        self.gate_proj = nn.Linear(emb, hid, bias=False)
        self.up_proj   = nn.Linear(emb, hid, bias=False)
        self.down_proj = nn.Linear(hid, emb, bias=False)
        self.silu = SiLU()

    def forward(self, x):
        gate = self.silu(self.gate_proj(x))      # gate_proj on x, then silu
        up   = self.up_proj(x)     # up_proj on x
        return self.down_proj(gate * up)      # down_proj on (gate * up)

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        x_sqr = torch.pow(x, 2)
        mean = x_sqr.mean(dim=-1, keepdim=True)
        rms_x = torch.sqrt(mean + self.eps)
        return x / rms_x * self.weight

def compute_rope_params(head_dim, theta_base=10_000, context_length=4096):
        assert head_dim % 2 == 0

        # one frequency per PAIR of dims → head_dim // 2 values
        inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2).float() / head_dim))  # hint: torch.arange(0, head_dim, 2).float()

        positions = torch.arange(0, context_length, 1).float()  # 0, 1, ..., context_length-1  (float)

        angles = positions[:, None] * inv_freq[None, :]  # outer product → shape (context_length, head_dim // 2)
        # hint: positions[:, None] * inv_freq[None, :]

        angles = torch.cat([angles, angles], dim=-1)  # → (context_length, head_dim)

        return torch.cos(angles), torch.sin(angles)

def apply_rope(x, cos, sin, offset=0):
    b, h, seq_len, head_dim = x.shape

    half = head_dim // 2

    x1 = x[...,:half]     # first half of the last dim
    x2 = x[...,half:]      # second half of the last dim
    rotated = torch.cat([-x2, x1], dim=-1)   # hint: (-x2, x1)

    cos = cos[offset:offset + seq_len, :]     # (seq_len, head_dim) — broadcasts over b and h
    sin = sin[offset:offset + seq_len, :]

    return x * cos + rotated * sin

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, d_out,  num_heads, num_kv_groups):
        super().__init__()
        assert d_out % num_heads == 0
        assert num_heads % num_kv_groups == 0
        self.d_out = d_out
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = d_out // num_heads
        self.group_size = num_heads // num_kv_groups

        self.q_proj = nn.Linear(d_in, num_heads * self.head_dim, bias=False)   # all query heads
        self.k_proj = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False)   # only num_kv_groups heads!
        self.v_proj = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False)   # same as k_proj
        self.o_proj = nn.Linear(d_out, d_out, bias=False)

    def forward(self, x, cos, sin, attention_mask=None, kv_cache=None, start_pos=0):
        b, T_new, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(b, T_new, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, T_new, self.num_kv_groups, self.head_dim).transpose(1, 2)
        v = v.view(b, T_new, self.num_kv_groups, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin, offset = start_pos)
        k = apply_rope(k, cos, sin, offset = start_pos)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)  # which dim is "tokens"?
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)  # saved BEFORE repeat_interleave
        T_total = k.shape[2]

        allowed = ~torch.triu(torch.ones(T_new, T_total, dtype=torch.bool, device=x.device),
                              diagonal=T_total - T_new + 1)

        if attention_mask is not None:
            allowed = allowed & attention_mask[:, None, None, :].bool()

        context_vec = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, enable_gqa=True)
        context_vec = context_vec.transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, T_new, self.d_out)
        context_vec = self.o_proj(context_vec)
        return context_vec, new_cache

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"])
        self.mlp = FeedForward(cfg)
        self.input_layernorm = RMSNorm(cfg["emb_dim"], eps=cfg["rms_eps"])
        self.post_attention_layernorm = RMSNorm(cfg["emb_dim"], eps=cfg["rms_eps"])

    def forward(self, x, cos, sin, attention_mask=None, kv_cache=None, start_pos=0):
        shortcut = x
        x = self.input_layernorm(x)
        x, new_cache = self.self_attn(x, cos, sin, attention_mask, kv_cache, start_pos)
        x = x + shortcut

        shortcut = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = x + shortcut
        return x, new_cache


class LlamaModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])

        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = RMSNorm(cfg["emb_dim"], eps=cfg["rms_eps"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        if cfg["tie_embeddings"]:
            self.out_head.weight = self.tok_emb.weight   # same tensor, not a copy

        cos, sin = compute_rope_params(
            head_dim=cfg["emb_dim"]//cfg["n_heads"],                  # emb_dim // n_heads
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"])

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, input_ids, attention_mask = None, kv_caches=None, start_pos=0, use_cache=False):
        batch_size, seq_len = input_ids.shape
        x = self.tok_emb(input_ids)

        if kv_caches is None:
            kv_caches = [None] * len(self.trf_blocks)  # first call: no history yet

        new_caches = []
        for block, layer_cache in zip(self.trf_blocks, kv_caches):
            x, layer_new_cache = block(x, self.cos, self.sin,
                                       attention_mask=attention_mask,
                                       kv_cache=layer_cache,  # this layer's old (k, v)
                                       start_pos=start_pos)
            new_caches.append(layer_new_cache)

        x = self.final_norm(x)
        logits = self.out_head(x)

        if use_cache:
            return logits, new_caches

        return logits
