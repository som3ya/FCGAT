#!/usr/bin/env python
"""Train Word2Vec (CBOW) on instruction tokens.

This script expects graphs created by `scripts/build_graph_dataset.py`.
Each graph has `x_tokens`: list[list[str]] where each inner list is a function.

Output: a Gensim Word2Vec model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import torch
from gensim.models import Word2Vec
from tqdm import tqdm


def _iter_sentences(graph_paths: List[Path]) -> Iterable[List[str]]:
    for p in graph_paths:
        data = torch.load(p)
        # `x_tokens` is a list of list-of-tokens per node/function.
        for sent in getattr(data, "x_tokens", []) or []:
            if sent:
                yield sent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs_raw_dir", type=Path, required=True)
    ap.add_argument("--out_model", type=Path, required=True)
    ap.add_argument("--vector_size", type=int, default=100)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--min_count", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    graph_paths = sorted(args.graphs_raw_dir.glob("*.pt"))
    if not graph_paths:
        raise FileNotFoundError(f"No .pt files found in {args.graphs_raw_dir}")

    # Gensim builds vocab from an iterable. We materialize to allow progress reporting.
    sentences = list(tqdm(_iter_sentences(graph_paths), desc="Collect sentences"))
    if not sentences:
        raise ValueError("No sentences found. Did you export function tokens correctly?")

    model = Word2Vec(
        sentences=sentences,
        vector_size=args.vector_size,
        window=args.window,
        sg=0,  # CBOW
        min_count=args.min_count,
        workers=args.workers,
        seed=args.seed,
        epochs=args.epochs,
    )

    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.out_model))
    print(f"[+] Saved Word2Vec model to {args.out_model}")


if __name__ == "__main__":
    main()
