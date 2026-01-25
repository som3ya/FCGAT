# IDAPython script
# Export Function Call Graph (FCG) as a DOT file.
#
# Usage (headless):
#   idat64 -A -S"export_fcg.py --out /path/to/out.dot" /path/to/sample.i64

import os

import ida_gdl
import idc


def _parse_out_path(argv, default_path="output_fcg/fcg.dot"):
    """Very small argv parser for IDA scripts."""
    if not argv:
        return default_path
    for i, a in enumerate(argv):
        if a == "--out" and i + 1 < len(argv):
            return argv[i + 1]
    return default_path


def main() -> None:
    out_path = _parse_out_path(getattr(idc, "ARGV", []))

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # NOTE:
    # - gen_simple_call_chart() writes a DOT file representing function calls.
    # - The last argument is a flag used by IDA; in the original artifact 0x2000 is used.
    ida_gdl.gen_simple_call_chart(out_path, "wait", "title", 0x2000)

    # Quit IDA after script completion.
    idc.qexit(0)


if __name__ == "__main__":
    main()
