import pytest
import torch
from torch import nn

from dl_lab.cnn import SmallCNN
from dl_lab.transformer import (
    MultiHeadAttention,
    SortingTransformer,
    scaled_dot_product_attention,
    sorting_batch,
)


def test_cnn_output_shape():
    assert SmallCNN()(torch.zeros(4, 1, 28, 28)).shape == (4, 10)


def test_cnn_overfits_one_batch():
    torch.manual_seed(0)
    model = SmallCNN()
    images, labels = torch.randn(16, 1, 28, 28), torch.randint(0, 10, (16,))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(60):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
    model.eval()
    assert (model(images).argmax(1) == labels).float().mean() > 0.9


def test_attention_weights_sum_to_one():
    x = torch.randn(2, 5, 8)
    _, weights = scaled_dot_product_attention(x, x, x)
    assert weights.sum(dim=-1).allclose(torch.ones(2, 5))


def test_attention_follows_matching_key():
    query = torch.tensor([[[10.0, 0.0]]])
    key = torch.tensor([[[10.0, 0.0], [0.0, 10.0]]])
    value = torch.tensor([[[1.0], [2.0]]])
    output, _ = scaled_dot_product_attention(query, key, value)
    assert output.item() == pytest.approx(1.0)


def test_multi_head_shapes():
    output, weights = MultiHeadAttention(16, 4)(torch.randn(3, 7, 16))
    assert output.shape == (3, 7, 16)
    assert weights.shape == (3, 4, 7, 7)


def test_heads_must_divide_d_model():
    with pytest.raises(ValueError):
        MultiHeadAttention(10, 4)


def test_sorting_batch_is_sorted_copy():
    tokens, target = sorting_batch(5)
    assert (target[:, 1:] >= target[:, :-1]).all()
    assert (tokens.sort(dim=1).values == target).all()


def test_transformer_output_shape():
    assert SortingTransformer()(torch.zeros(2, 8, dtype=torch.long)).shape == (2, 8, 10)
