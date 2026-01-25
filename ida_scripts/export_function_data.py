# IDAPython script
# Export per-function disassembly tokens.
#
# Usage (headless):
#   idat64 -A -S"export_function_data.py --out /path/to/out.pkl" /path/to/sample.i64

import os
import pickle
import re

import idautils
import idc


def _parse_out_path(argv, default_path="output_functiondata/function_data.pkl"):
    if not argv:
        return default_path
    for i, a in enumerate(argv):
        if a == "--out" and i + 1 < len(argv):
            return argv[i + 1]
    return default_path


# Replace hex/decimal constants with a placeholder to reduce vocab size.
_HEX_PATTERNS = [
    re.compile(r"0x[0-9A-Fa-f]+"),
    re.compile(r"\b[0-9A-Fa-f]+h\b"),
    # IDA sometimes prints offsets as plain hex without suffix; keep this conservative:
    re.compile(r"\b[0-9A-Fa-f]{6,}\b"),
]
_DEC_PATTERN = re.compile(r"\b\d+\b")


def _tokenize_disasm_line(line: str) -> str:
    """Tokenize a disassembly line into a single 'instruction token'.

    We follow the paper's intent: remove comments, normalize constants,
    and reduce vocabulary size.

    The output is a single token string (whitespace joined by '_').
    """
    # Remove comments after ';'
    if ";" in line:
        line = line.split(";", 1)[0]

    line = line.replace(",", " ")
    line = line.strip()

    for pat in _HEX_PATTERNS:
        line = pat.sub("N", line)
    line = _DEC_PATTERN.sub("N", line)

    # Collapse whitespace and join to one token
    parts = line.split()
    return "_".join(parts)


def main() -> None:
    out_path = _parse_out_path(getattr(idc, "ARGV", []))
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    offsets = []
    instructions = {}
    func_dict = {}

    for func_ea in idautils.Functions():
        offsets.append(func_ea)
        func_name = idc.get_func_name(func_ea)
        func_dict[func_name] = func_ea

        inst_tokens = []
        for ea in idautils.FuncItems(func_ea):
            disasm = idc.generate_disasm_line(ea, 0) or ""
            tok = _tokenize_disasm_line(disasm)
            if tok:
                inst_tokens.append(tok)

        instructions[func_ea] = inst_tokens

    function_data = {
        "offsets": offsets,
        "instructions": instructions,
        "function": func_dict,
    }

    with open(out_path, "wb") as f:
        pickle.dump(function_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    idc.qexit(0)


if __name__ == "__main__":
    main()
