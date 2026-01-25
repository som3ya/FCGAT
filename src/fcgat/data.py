from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def load_graphs(data_dir: Path) -> List:
    """Load all `.pt` graph files from a directory."""
    paths = sorted(data_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No .pt files found in {data_dir}")
    graphs = [torch.load(p) for p in paths]
    return graphs


def load_label_map(data_dir: Path) -> Optional[dict]:
    """Load `id_to_label.json` if present."""
    p = data_dir / "id_to_label.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


@dataclass
class Split:
    train: List
    test: List


def train_test_split_graphs(
    graphs: List,
    test_size: float = 0.2,
    seed: int = 0,
    stratify: bool = True,
) -> Split:
    y = np.array([int(g.y) for g in graphs])
    strat = y if stratify else None
    tr, te = train_test_split(graphs, test_size=test_size, random_state=seed, stratify=strat)
    return Split(train=tr, test=te)


def compute_metrics(y_true: List[int], y_pred: List[int]) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": acc,
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
    }
