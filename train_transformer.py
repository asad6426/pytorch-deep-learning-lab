"""Train SortingTransformer on freshly generated sequences, with a held-out seed."""
import json
import time
from pathlib import Path

import torch
from torch import nn

from dl_lab.transformer import SortingTransformer, sorting_batch

STEPS, BATCH_SIZE, SEED = 3000, 128, 42


@torch.no_grad()
def evaluate(model, batches=20, batch_size=512):
    model.eval()
    generator = torch.Generator().manual_seed(SEED + 1)
    token_hits = sequence_hits = sequences = 0
    for _ in range(batches):
        tokens, target = sorting_batch(batch_size, generator=generator)
        predicted = model(tokens).argmax(dim=-1)
        token_hits += (predicted == target).sum().item()
        sequence_hits += (predicted == target).all(dim=1).sum().item()
        sequences += batch_size
    return token_hits / (sequences * target.size(1)), sequence_hits / sequences


def main():
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    model = SortingTransformer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    started, history = time.time(), []
    for step in range(1, STEPS + 1):
        model.train()
        tokens, target = sorting_batch(BATCH_SIZE, generator=generator)
        loss = loss_fn(model(tokens).transpose(1, 2), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            token_accuracy, sequence_accuracy = evaluate(model)
            row = {"step": step, "train_loss": round(loss.item(), 4),
                   "token_accuracy": round(token_accuracy, 4),
                   "sequence_accuracy": round(sequence_accuracy, 4)}
            history.append(row)
            print(row, flush=True)

    report = {
        "task": "sort 8 random digits",
        "parameters": sum(p.numel() for p in model.parameters()),
        "steps": STEPS,
        "seed": SEED,
        "device": "cpu",
        "train_seconds": round(time.time() - started),
        "final_token_accuracy": history[-1]["token_accuracy"],
        "final_sequence_accuracy": history[-1]["sequence_accuracy"],
        "history": history,
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/transformer.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
