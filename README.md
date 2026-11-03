# PyTorch Deep Learning Lab

[![Train and test](https://github.com/asad6426/pytorch-deep-learning-lab/actions/workflows/train.yml/badge.svg)](https://github.com/asad6426/pytorch-deep-learning-lab/actions/workflows/train.yml)

Two small models I use to teach deep learning: a CNN trained on a real image dataset, and a Transformer encoder written from the attention equation upward. Both train on a CPU, so students can run them without a GPU.

## The models

- **[`dl_lab/cnn.py`](dl_lab/cnn.py):** two convolution blocks and a linear classifier, about 50,000 parameters.
- **[`dl_lab/transformer.py`](dl_lab/transformer.py):** scaled dot-product attention, multi-head attention and encoder layers written by hand instead of using `nn.TransformerEncoder`, about 100,000 parameters.

## Results

Every number comes from [this GitHub Actions run](https://github.com/asad6426/pytorch-deep-learning-lab/actions/runs/41676540817) on a CPU with seed 42 and PyTorch 2.15. The full output is in [`results/`](results/).

### CNN on Fashion-MNIST

**90.25% test accuracy** after 5 epochs, in about 5 minutes. This is the final epoch's result. Epoch 3 reached 90.76%, but picking the best epoch by test score would leak the test set into model selection.

| Strongest classes | Accuracy | Weakest classes | Accuracy |
| --- | --- | --- | --- |
| Trouser | 99.0% | Shirt | 59.4% |
| Bag | 98.6% | Dress | 85.0% |
| Sandal | 97.8% | Coat | 85.4% |

Shirt is the clear weak spot, which makes a good discussion of why one headline accuracy figure is not enough.

### Transformer sorting 8 random digits

**99.99% of whole sequences correct** on 10,240 freshly generated sequences, after 3,000 steps in 48 seconds.

Training was not smooth. Sequence accuracy fell to 86.7% at step 2,000 before recovering, which shows why one checkpoint can mislead.

## Limits

- Sorting digits is a toy task chosen to show attention at work, not a language model. The test sequences come from the same random distribution as the training data.
- The CNN is deliberately small and has no data augmentation, so 90% is a teaching baseline and not a competitive Fashion-MNIST score.
- There was one training run per model, so the numbers do not show variation across seeds.

## Run it

```bash
pip install -r requirements.txt
python -m pytest -q
python train_transformer.py
python train_cnn.py
```

`train_cnn.py` downloads Fashion-MNIST the first time it runs. The 8 tests cover output shapes, attention behaviour and whether the CNN can overfit one batch.

Built with Claude Code.
