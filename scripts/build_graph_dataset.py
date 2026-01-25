#!/usr/bin/env python
"""Build a PyTorch Geometric graph dataset from IDA exports.

Inputs:
- FCG DOT files exported by `ida_scripts/export_fcg.py`
- function token PKL files exported by `ida_scripts/export_function_data.py`

Output:
- one `.pt` file per sample (torch.save of torch_geometric.data.Data)
  with fields:
    - edge_index: reversed FCG edges (called -> caller)
    - x_tokens: list[list[str]]   (tokenized instructions per function)
    - func_name: list[str]        (function names aligned with nodes)
    - y: int64 label id
    - sha256: str sample id

The next step is to train Word2Vec on `x_tokens` and embed them into tensors.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import networkx as nx
from torch_geometric.utils import from_networkx


def _read_dot(dot_path: Path) -> nx.DiGraph:
    # IDA's DOT sometimes results in MultiDiGraph; we normalize to DiGraph.
    g = nx.drawing.nx_pydot.read_dot(str(dot_path))
    g = nx.DiGraph(g)

    # IDA sometimes creates a stray newline node.
    if "\\n" in g:
        g.remove_node("\\n")

    # Some nodes may have a 'color' attribute; it is irrelevant for us.
    for n, attrs in list(g.nodes(data=True)):
        if isinstance(attrs, dict) and "color" in attrs:
            attrs.pop("color", None)

    return g


def _node_label(g: nx.DiGraph, node) -> str:
    attrs = g.nodes[node] if node in g.nodes else {}
    label = attrs.get("label", str(node))

    # DOT parsers often keep quotes.
    if isinstance(label, str):
        label = label.strip().strip('"')

    # IDA labels may include extra formatting; keep it simple.
    return label


def _load_function_data(pkl_path: Path) -> Dict:
    import pickle

    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _build_label_encoder(labels_csv: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    df = pd.read_csv(labels_csv)
    if "sha256" not in df.columns or "label" not in df.columns:
        raise ValueError(
            "labels_csv must contain columns 'sha256' and 'label'. "
            "See examples/labels_example.csv"
        )

    labels = sorted(df["label"].dropna().unique().tolist())
    label_to_id = {lbl: i for i, lbl in enumerate(labels)}
    id_to_label = {i: lbl for lbl, i in label_to_id.items()}
    return label_to_id, id_to_label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fcg_dir", type=Path, required=True, help="Directory containing *.dot")
    ap.add_argument(
        "--functiondata_dir",
        type=Path,
        required=True,
        help="Directory containing *.pkl from export_function_data.py",
    )
    ap.add_argument(
        "--labels_csv",
        type=Path,
        required=True,
        help="CSV with columns: sha256,label",
    )
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--skip_missing", action="store_true", help="Skip samples with missing files")

    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    label_to_id, id_to_label = _build_label_encoder(args.labels_csv)
    (args.out_dir / "label_to_id.json").write_text(json.dumps(label_to_id, indent=2), encoding="utf-8")
    (args.out_dir / "id_to_label.json").write_text(json.dumps(id_to_label, indent=2), encoding="utf-8")

    df_labels = pd.read_csv(args.labels_csv)
    sha_to_label = dict(zip(df_labels["sha256"].astype(str), df_labels["label"].astype(str)))

    dot_files = sorted(args.fcg_dir.glob("*.dot"))
    if not dot_files:
        raise FileNotFoundError(f"No .dot files found in {args.fcg_dir}")

    num_written = 0
    num_skipped = 0

    for dot_path in dot_files:
        sha256 = dot_path.stem
        pkl_path = args.functiondata_dir / f"{sha256}.pkl"

        if sha256 not in sha_to_label:
            num_skipped += 1
            continue

        if not pkl_path.exists():
            if args.skip_missing:
                num_skipped += 1
                continue
            raise FileNotFoundError(f"Missing function data: {pkl_path}")

        label_str = sha_to_label[sha256]
        y = label_to_id[label_str]

        g = _read_dot(dot_path)
        fd = _load_function_data(pkl_path)
        func_dict = fd.get("function", {})  # name -> start_ea
        inst_dict = fd.get("instructions", {})  # start_ea -> [tokens]

        # Keep only nodes whose label matches a function name found by IDA.
        keep_nodes = []
        func_names = []
        x_tokens: List[List[str]] = []

        for node in list(g.nodes()):
            name = _node_label(g, node)
            if name not in func_dict:
                continue
            keep_nodes.append(node)

        if not keep_nodes:
            num_skipped += 1
            continue

        g2 = g.subgraph(keep_nodes).copy()

        # Build features aligned with the NetworkX node order.
        node_list = list(g2.nodes())
        for node in node_list:
            name = _node_label(g2, node)
            func_names.append(name)
            start_ea = func_dict.get(name)
            toks = inst_dict.get(start_ea, []) if start_ea is not None else []
            if toks is None:
                toks = []
            x_tokens.append(list(toks))

        data = from_networkx(g2)

        # Reverse FCG edges (called -> caller) as described in the paper.
        data.edge_index = data.edge_index.flip(0).contiguous()

        data.x_tokens = x_tokens
        data.func_name = func_names
        data.y = torch.tensor(y, dtype=torch.long)
        data.sha256 = sha256

        out_path = args.out_dir / f"{sha256}.pt"
        torch.save(data, out_path)
        num_written += 1

    print(f"[+] Wrote {num_written} graphs to {args.out_dir}")
    print(f"[i] Skipped {num_skipped} graphs (missing label / missing files / empty graph)")


if __name__ == "__main__":
    main()
