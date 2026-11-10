#!/usr/bin/env python3
"""Audit P2-B merged libraries and build one fresh IR overlay per library.

This script must run inside the user's original project with p1b_audit.py,
p1d_build_ir.py, p1c.retrieval, and p1d.ir available. No Qwen calls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library-root", default="p2b_libraries")
    p.add_argument("--ir-root", default="p2b_ir_libraries")
    p.add_argument("--holdout-manifest", action="append", required=True)
    p.add_argument("--sizes", default="300,500,1000")
    args = p.parse_args()
    root = Path.cwd().resolve()
    lib_root = Path(args.library_root).resolve()
    ir_root = Path(args.ir_root).resolve()
    sizes = [int(n.strip()) for n in args.sizes.split(",")]
    names = [f"p2b_{mode}_{n}_v1"
             for mode in ("standard", "balanced") for n in sizes]
    libraries = [lib_root / name for name in names]
    overlays = [ir_root / f"p2b_{mode}_{n}_ir_v1"
                for mode in ("standard", "balanced") for n in sizes]
    for name in ("p1b_audit.py", "p1d_build_ir.py"):
        if not (root / name).is_file():
            raise FileNotFoundError("Missing required original script: " + name)
    for lib in libraries:
        if not (lib / "library_manifest.json").is_file():
            raise FileNotFoundError("Missing P2-B library: " + str(lib))
    for overlay in overlays:
        if overlay.exists():
            raise FileExistsError("Refuse to overwrite IR overlay: " + str(overlay))
    for hp in args.holdout_manifest:
        if not Path(hp).is_file():
            raise FileNotFoundError("Missing holdout manifest: " + hp)

    def checked(cmd: list[str]):
        print("RUN:", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=root, check=True)

    # First audit and validate all merged source libraries before building overlays.
    for lib in libraries:
        cmd = [sys.executable, "p1b_audit.py", "--library", str(lib)]
        for hp in args.holdout_manifest:
            cmd.extend(["--holdout-manifest", hp])
        checked(cmd)

    from p1c.retrieval import LocalHistory
    from p1d.ir import IRHistory
    from p1b.core import read_json
    parsed = []
    for lib in libraries:
        obj = LocalHistory(lib)
        if obj.meta.get("n_experiences_per_role") != int(lib.name.split("_")[-2]):
            raise RuntimeError("LocalHistory count mismatch in " + str(lib))
        for hp in args.holdout_manifest:
            obj.assert_independent(read_json(hp))
        parsed.append(obj)
        print("LOCAL HISTORY OK:", lib, flush=True)

    ir_root.mkdir(parents=True, exist_ok=True)
    output = []
    for lib, overlay, history in zip(libraries, overlays, parsed):
        checked([sys.executable, "p1d_build_ir.py", "--library", str(lib),
                 "--output-dir", str(overlay)])
        if not (overlay / "manifest.json").is_file() and not (
                overlay / "ir_manifest.json").is_file():
            # Do not assume the exact IR manifest name in the P1-D implementation;
            # construct IRHistory, which enforces the actual local schema.
            print("NOTE: IR manifest filename differs from common names", flush=True)
        ir = IRHistory(overlay, history)
        output.append({"library": str(lib), "ir_overlay": str(overlay),
                       "ir_manifest_sha256": getattr(ir, "manifest_sha256", None)})
        print("IR OVERLAY OK:", overlay, flush=True)
    print(json.dumps({"ok": True, "audited_libraries": len(libraries),
                      "verified_ir_overlays": len(overlays), "pairs": output},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as ex:
        print("STOPPED:", ex, file=sys.stderr)
        sys.exit(2)
