#!/usr/bin/env bash
set -euo pipefail

# Batch export FCG DOT and function token PKL for all IDB/I64 files.
#
# Example:
#   IDA=/opt/ida/idat64 \
#   IDB_DIR=/data/idb \
#   OUT_DIR=/data/exports \
#   ./scripts/ida_export_all.sh

: "${IDA:?Set IDA (path to idat64/idat.exe wrapper)}"
: "${IDB_DIR:?Set IDB_DIR (directory containing .idb/.i64)}"
: "${OUT_DIR:?Set OUT_DIR (output root)}"

FCG_DIR="$OUT_DIR/fcg_dot"
FUNC_DIR="$OUT_DIR/functiondata_pkl"
mkdir -p "$FCG_DIR" "$FUNC_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

for f in "$IDB_DIR"/* ; do
  base="$(basename "$f")"
  name="${base%%.*}"
  echo "[+] Exporting: $name"

  "$IDA" -A -S"$REPO_ROOT/ida_scripts/export_fcg.py --out $FCG_DIR/$name.dot" "$f"
  "$IDA" -A -S"$REPO_ROOT/ida_scripts/export_function_data.py --out $FUNC_DIR/$name.pkl" "$f"

done

echo "[+] Done. Outputs:"
echo "  - $FCG_DIR"
echo "  - $FUNC_DIR"
