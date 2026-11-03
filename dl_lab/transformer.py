"""A Transformer encoder written from the attention equation upward."""
import math

import torch
from torch import nn


def scaled_dot_product_attention(query, key, value):
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V."""
    scores = query @ key.transpose(-2, -1) / math.sqrt(key.size(-1))
    weights = scores.softmax(dim=-1)
    return weights @ value, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.to_qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, length, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        split = lambda t: t.view(batch, length, self.num_heads, self.d_head).transpose(1, 2)
        attended, weights = scaled_dot_product_attention(split(q), split(k), split(v))
        merged = attended.transpose(1, 2).reshape(batch, length, -1)
        return self.out(merged), weights


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attended, weights = self.attention(self.norm1(x))
        x = x + attended
        return x + self.feed_forward(self.norm2(x)), weights


class SortingTransformer(nn.Module):
    """Reads a digit sequence and predicts the sorted digit for every position."""

    def __init__(self, vocab_size=10, seq_len=8, d_model=64, num_heads=4, num_layers=2):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(seq_len, d_model)
        self.layers = nn.ModuleList(EncoderLayer(d_model, num_heads) for _ in range(num_layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens):
        positions = torch.arange(tokens.size(1), device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        for layer in self.layers:
            x, _ = layer(x)
        return self.head(self.norm(x))


def sorting_batch(batch_size, seq_len=8, vocab_size=10, generator=None):
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), generator=generator)
    return tokens, tokens.sort(dim=1).values
