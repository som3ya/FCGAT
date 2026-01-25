from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .model import FCGAT, FCGATHParams


def _load_checkpoint(path: Path):
    ckpt = torch.load(path, map_location="cpu")
    if "model_state" not in ckpt:
        raise ValueError("checkpoint must contain model_state")
    return ckpt


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict and explain a single graph")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--graph_pt", type=Path, required=True)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--step", type=int, default=None, help="which Set2Set step to use (default: last)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()

    ckpt = _load_checkpoint(args.checkpoint)
    hparams = FCGATHParams(**ckpt.get("hparams", {}))
    num_classes = int(ckpt.get("num_classes"))

    model = FCGAT(num_classes=num_classes, hparams=hparams)
    model.load_state_dict(ckpt["model_state"])
    model.to(torch.device(args.device))

    data = torch.load(args.graph_pt)
    data = data.to(torch.device(args.device))

    result = model.explain_single(data, top_k=args.top_k, step=args.step)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
