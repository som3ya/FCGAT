# Contributing

Thanks for your interest in contributing.

## Rules

- **Do not** attach or commit malware binaries, unpacked samples, or proprietary datasets.
- If you need to share a problematic sample, share only its hash and metadata.
- Keep changes small and well documented.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[train]
```

(Install PyTorch / PyG according to your environment.)
