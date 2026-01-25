from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from .data import compute_metrics, load_graphs, train_test_split_graphs
from .model import FCGAT, FCGATHParams


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model: FCGAT, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(batch)
        y = batch.y.view(-1).long()
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * int(batch.num_graphs)
    return total_loss / max(1, len(loader.dataset))


@torch.no_grad()
def evaluate(model: FCGAT, loader: DataLoader, device: torch.device) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []

    for batch in tqdm(loader, desc="eval", leave=False):
        batch = batch.to(device)
        logits, _ = model(batch)
        y = batch.y.view(-1).long()
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * int(batch.num_graphs)

        pred = logits.argmax(dim=-1).detach().cpu().tolist()
        true = y.detach().cpu().tolist()
        y_pred.extend(pred)
        y_true.extend(true)

    metrics = compute_metrics(y_true, y_pred)
    return total_loss / max(1, len(loader.dataset)), metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Train FCGAT on embedded graph dataset (.pt)")
    ap.add_argument("--data_dir", type=Path, required=True, help="Directory with embedded graph .pt files")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=700)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)

    # Model hparams
    ap.add_argument("--in_dim", type=int, default=100)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=3)
    ap.add_argument("--set2set_steps", type=int, default=4)
    ap.add_argument("--dropout_p", type=float, default=0.5)
    ap.add_argument("--edge_dropout_p", type=float, default=0.5)

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(args.seed)

    device = torch.device(args.device)

    graphs = load_graphs(args.data_dir)
    split = train_test_split_graphs(graphs, test_size=args.test_size, seed=args.seed, stratify=True)

    # Determine num_classes from labels present.
    labels = sorted({int(g.y) for g in graphs})
    num_classes = len(labels)

    hparams = FCGATHParams(
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        set2set_steps=args.set2set_steps,
        dropout_p=args.dropout_p,
        edge_dropout_p=args.edge_dropout_p,
    )
    model = FCGAT(num_classes=num_classes, hparams=hparams).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(split.train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(split.test, batch_size=args.batch_size, shuffle=False)

    best_f1 = -1.0
    best_path = args.out_dir / "model.pt"

    history = []

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        te_loss, te_metrics = evaluate(model, test_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "test_loss": te_loss,
            **te_metrics,
        }
        history.append(row)

        if te_metrics["f1_macro"] > best_f1:
            best_f1 = te_metrics["f1_macro"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "hparams": hparams.__dict__,
                    "num_classes": num_classes,
                },
                best_path,
            )

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"[epoch {epoch:4d}] train_loss={tr_loss:.4f}  test_loss={te_loss:.4f}  "
                f"acc={te_metrics['accuracy']:.4f}  f1={te_metrics['f1_macro']:.4f}"
            )

    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[+] Saved training history to {args.out_dir/'history.json'}")
    print(f"[+] Best checkpoint: {best_path} (best f1_macro={best_f1:.4f})")


if __name__ == "__main__":
    main()
