# IDA Pro scripts

These scripts are **IDAPython** scripts and must be executed inside **IDA Pro**.
They are used to export:

- Function Call Graph (FCG) as a DOT file
- Per-function disassembly token sequences (pickled)

> ⚠️ Do not run IDA on untrusted samples on your host OS. Use an isolated VM.

---

## Scripts

- `export_fcg.py`
  - exports a DOT file (IDA `ida_gdl.gen_simple_call_chart`)
- `export_function_data.py`
  - exports a pickle containing:
    - function start addresses
    - function name -> start address mapping
    - tokenized instruction strings for each function

---

## Headless execution example

IDA allows passing arguments to `-S"script.py arg1 arg2 ..."`.
The scripts below accept output path arguments.

```bash
# Example paths
IDA=/path/to/idat64
IDB=path/to/sample.i64
OUT_FCG=output/fcg/sample.dot
OUT_FUNC=output/functiondata/sample.pkl

# Export FCG
$IDA -A -S"ida_scripts/export_fcg.py --out $OUT_FCG" $IDB

# Export function tokens
$IDA -A -S"ida_scripts/export_function_data.py --out $OUT_FUNC" $IDB
```

---

## Output formats

### FCG DOT
A standard DOT file readable by `pydot`/`networkx`.

### Function token pickle
A Python dict serialized via `pickle`:

```python
{
  "offsets": [0x401000, 0x401120, ...],
  "instructions": {
     0x401000: ["push_ebp", "mov_ebp_esp", ...],
     ...
  },
  "function": {
     "sub_401000": 0x401000,
     "CreateFileA": 0x...,
     ...
  }
}
```

---

## Tokenization rules

The tokenization in `export_function_data.py` is designed to reduce vocabulary size:

- remove commas
- remove comments after `;`
- replace hex immediates / offsets with `N`
- join whitespace with `_` so an instruction becomes a single token

You can adjust these rules to better match your analysis toolchain.
