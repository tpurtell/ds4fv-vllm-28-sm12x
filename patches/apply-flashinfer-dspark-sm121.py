#!/usr/bin/env python3
"""Add the SM121 sparse-MLA width required by native DSpark.

DeepSeek V4's DSpark draft expands the 128-token SWA window to 192 entries
at draft depth five.  FlashInfer's DSV4 kernel consumes candidates in 64-wide
chunks, but the pinned package omits the otherwise legal three-chunk template
from its Python capability table and C++ dispatch switch.  The package also
ships an AOT module under the unversioned source-module name, so give the
adapted sources their own JIT name rather than silently loading that old .so.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    source = path.read_text()
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: {label} expected exactly one source anchor, found {count}"
        )
    path.write_text(source.replace(old, new, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("site_packages", type=Path)
    args = parser.parse_args()

    python_dispatch = (
        args.site_packages / "flashinfer/mla/_sparse_mla_sm120.py"
    )
    cpp_dispatch = (
        args.site_packages
        / "flashinfer/data/csrc/sparse_mla_sm120_decode_dsv4.cu"
    )
    jit_generator = args.site_packages / "flashinfer/jit/mla.py"

    heads = (8, 16, 32, 64, 128)
    old_entries = "".join(
        f"        ({head}, 128),\n"
        f"        ({head}, 512),\n"
        f"        ({head}, 1024),\n"
        for head in heads
    )
    new_entries = "".join(
        f"        ({head}, 128),\n"
        f"        ({head}, 192),\n"
        f"        ({head}, 512),\n"
        f"        ({head}, 1024),\n"
        for head in heads
    )
    replace_once(
        python_dispatch,
        "_DECODE_DSV4_DISPATCH = frozenset(\n    {\n"
        + old_entries
        + "    }\n)\n",
        "_DECODE_DSV4_DISPATCH = frozenset(\n    {\n"
        + new_entries
        + "    }\n)\n",
        "Python DSV4 K192 dispatch table",
    )

    for head in heads:
        replace_once(
            cpp_dispatch,
            f"  DSV4_DISPATCH({head}, 128)\n"
            f"  DSV4_DISPATCH({head}, 512)",
            f"  DSV4_DISPATCH({head}, 128)\n"
            f"  DSV4_DISPATCH({head}, 192)\n"
            f"  DSV4_DISPATCH({head}, 512)",
            f"C++ DSV4 H{head}/K192 instantiation",
        )

    replace_once(
        python_dispatch,
        "``topk`` must be one of {128, 512, 1024}",
        "``topk`` must be one of {128, 192, 512, 1024}",
        "Python DSV4 top-k documentation",
    )
    replace_once(
        cpp_dispatch,
        "// TOPK ∈ {128, 512, 1024}.",
        "// TOPK ∈ {128, 192, 512, 1024}.",
        "C++ DSV4 top-k documentation",
    )
    replace_once(
        jit_generator,
        '        "sparse_mla_sm120",\n',
        '        "sparse_mla_sm120_ds4fv_k192_v1",\n',
        "adapted sparse-MLA JIT module name",
    )
    print("FlashInfer SM121 DSV4 sparse-MLA K192 dispatch installed")


if __name__ == "__main__":
    main()
