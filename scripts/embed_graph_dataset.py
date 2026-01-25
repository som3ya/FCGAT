#!/usr/bin/env python
"""Embed function token sequences into vectors using a trained Word2Vec model.

Input graphs come from `scripts/build_graph_dataset.py`.
Output graphs are saved as `.pt` with:
- x: FloatTensor[num_nodes, vector_size]
- y: LongTensor[]
- func_name: list[str]
- sha256: str

By default, `x_tokens` are removed from the embedded graphs to keep files small.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
from gensim.models import Word2Vec
from tqdm import tqdm


def _embed_sentence(model: Word2Vec, tokens: List[str], dim: int) -> np.ndarray:
    """Average instruction vectors in a function."""
    vecs = []
    for tok in tokens:
        if tok in model.wv:
            vecs.append(model.wv[tok])
    if not vecs:
        return np.zeros((dim,), dtype=np.float32)
    return np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs_raw_dir", type=Path, required=True)
    ap.add_argument("--word2vec_model", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--keep_tokens", action="store_true")

    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = Word2Vec.load(str(args.word2vec_model))
    dim = int(model.vector_size)

    graph_paths = sorted(args.graphs_raw_dir.glob("*.pt"))
    if not graph_paths:
        raise FileNotFoundError(f"No .pt files found in {args.graphs_raw_dir}")

    for p in tqdm(graph_paths, desc="Embedding graphs"):
        data = torch.load(p)
        x_tokens = getattr(data, "x_tokens", None)
        if x_tokens is None:
            raise ValueError(
                f"Graph {p} has no x_tokens. Did you run build_graph_dataset.py?"
            )

        x = np.stack([_embed_sentence(model, toks, dim) for toks in x_tokens], axis=0)
        data.x = torch.tensor(x, dtype=torch.float32)

        # Optionally drop raw tokens to reduce disk usage.
        if not args.keep_tokens:
            try:
                delattr(data, "x_tokens")
            except Exception:
                pass

        out_path = args.out_dir / p.name
        torch.save(data, out_path)

    print(f"[+] Embedded graphs written to {args.out_dir}")


if __name__ == "__main__":
    main()
