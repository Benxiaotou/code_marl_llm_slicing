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



def read_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def assert_v1_v2_content_parity(v1: Path, v2: Path):
    """Require identical case selection, original observations and teacher actions.

    Only the dataset/joint/role IDs may differ. Source originals stay recorded
    in private_offline/source_case_id_map.jsonl, never inside online role rows.
    """
    mf = read_json(v2 / "library_manifest.json")
    prefix = str(mf["dataset_id"]) + "::"
    old = read_jsonl(v1 / "private_offline" / "selected_joint_records.jsonl")
    new = read_jsonl(v2 / "private_offline" / "selected_joint_records.jsonl")
    mapping = read_jsonl(v2 / "private_offline" / "source_case_id_map.jsonl")
    if not (len(old) == len(new) == len(mapping) == mf["n_experiences_per_role"]):
        raise RuntimeError("v1/v2 joint record or mapping count mismatch: " + v2.name)
    for a, b, origin in zip(old, new, mapping):
        if (a["joint_id"] != origin["original_joint_id"] or
                a["dataset_id"] != origin["original_dataset_id"] or
                b["joint_id"] != origin["composite_joint_id"] or
                not b["joint_id"].startswith(prefix)):
            raise RuntimeError("v1/v2 source identity mismatch: " + v2.name)
        reconstructed = dict(b)
        reconstructed["joint_id"] = a["joint_id"]
        reconstructed["dataset_id"] = a["dataset_id"]
        if reconstructed != a:
            raise RuntimeError("v1/v2 offline joint content changed: " + v2.name)
    old_idx = read_json(v1 / "private_offline" / "selection_index.json")
    new_idx = read_json(v2 / "private_offline" / "selection_index.json")
    if len(old_idx) != len(new_idx):
        raise RuntimeError("v1/v2 selection index count changed: " + v2.name)
    for a, b in zip(old_idx, new_idx):
        restored = dict(b)
        restored["joint_id"] = a["joint_id"]
        if restored != a:
            raise RuntimeError("v1/v2 selection index changed: " + v2.name)
    for role in ("agent_k1", "agent_k2", "agent_k3", "agent_g"):
        a = read_jsonl(v1 / "role_libraries" / (role + ".jsonl"))
        b = read_jsonl(v2 / "role_libraries" / (role + ".jsonl"))
        if len(a) != len(old) or len(b) != len(old):
            raise RuntimeError("v1/v2 role count changed: " + role)
        for pos, (before, after) in enumerate(zip(a, b)):
            if (before["experience_id"] != old[pos]["joint_id"] or
                    after["experience_id"] != new[pos]["joint_id"]):
                raise RuntimeError("v1/v2 role ID mapping changed: " + role)
            orig = dict(after)
            orig["experience_id"] = before["experience_id"]
            if orig != before:
                raise RuntimeError("v1/v2 role observation or action changed: " + role)
    print("V1/V2 CONTENT PARITY OK:", v2.name, flush=True)


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
    names = [f"p2b_{mode}_{n}_v2"
             for mode in ("standard", "balanced") for n in sizes]
    libraries = [lib_root / name for name in names]
    overlays = [ir_root / f"p2b_{mode}_{n}_ir_v2"
                for mode in ("standard", "balanced") for n in sizes]
    for name in ("p1b_audit.py", "p1d_build_ir.py"):
        if not (root / name).is_file():
            raise FileNotFoundError("Missing required original script: " + name)
    for lib in libraries:
        if not (lib / "library_manifest.json").is_file():
            raise FileNotFoundError("Missing P2-B library: " + str(lib))
        v1 = lib.with_name(lib.name[:-3] + "_v1")
        if not (v1 / "library_manifest.json").is_file():
            raise FileNotFoundError("Missing original v1 library for parity check: " + str(v1))
    for overlay in overlays:
        if overlay.exists():
            raise FileExistsError("Refuse to overwrite IR overlay: " + str(overlay))
    for hp in args.holdout_manifest:
        if not Path(hp).is_file():
            raise FileNotFoundError("Missing holdout manifest: " + hp)

    def checked(cmd: list[str]):
        print("RUN:", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=root, check=True)

    # Confirm v2 changes nothing except case identifiers and provenance sidecar.
    for lib in libraries:
        assert_v1_v2_content_parity(lib.with_name(lib.name[:-3] + "_v1"), lib)

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
