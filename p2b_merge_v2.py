#!/usr/bin/env python3
"""P2-B v2 offline composite libraries with consistent namespace IDs.

Run from the original project root. This script does not call any LLM, MAPPO,
or the evaluation environment. Default mode is read-only (--execute writes).

Requires original P1-B libraries (library_manifest.json + private source +
role_libraries). Outputs must still pass the project's p1b_audit.py and IR
compatibility checks before use in experiments.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

ROLES = ("agent_k1", "agent_k2", "agent_k3", "agent_g")
ROLE_KEYS = {
    "schema", "role", "experience_id", "observation_240",
    "own_teacher_ratios", "own_teacher_env_action", "source",
}
DEFAULT_OLD = "p1b_libraries/train_seed2601_v1"
DEFAULT_NEW = "p2b_libraries/p2b_train_seed2602_candidate_v1"
DEFAULT_OUTPUT = "p2b_libraries"
DEFAULT_SIZES = (300, 500, 1000)


class InputError(ValueError):
    pass


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for piece in iter(lambda: f.read(1024 * 1024), b""):
            h.update(piece)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    result = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                raise InputError(f"Blank JSONL line at {path}:{line_no}")
            try:
                record = json.loads(line)
            except Exception as ex:
                raise InputError(f"Invalid JSONL at {path}:{line_no}: {ex}") from ex
            if not isinstance(record, dict):
                raise InputError(f"Not an object at {path}:{line_no}")
            result.append(record)
    return result


def emit_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def emit_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(canonical_json(rec) + "\n")


def obs_signature(record: dict) -> str:
    obs = record.get("observations")
    if not isinstance(obs, dict) or any(r not in obs for r in ROLES):
        raise InputError("Missing four role observations in private joint record")
    return sha_bytes(canonical_json([obs[r] for r in ROLES]).encode("utf-8"))


def finite_vector(v: Any, n: int, what: str) -> None:
    if not isinstance(v, list) or len(v) != n:
        raise InputError(f"{what}: expected list[{n}]")
    for x in v:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            raise InputError(f"{what}: non-finite or nonnumeric entry")


def quality(rec: dict) -> float:
    q = rec.get("selection_quality_score")
    if not isinstance(q, (int, float)) or isinstance(q, bool) or not math.isfinite(q):
        raise InputError("Missing finite selection_quality_score in selected private record")
    return float(q)


def stratum(rec: dict) -> str:
    if "selection_stratum" not in rec:
        raise InputError("Missing selection_stratum from original P1-B selection")
    return canonical_json(rec["selection_stratum"])


class SourceLibrary:
    def __init__(self, path: Path, label: str):
        self.path = path.resolve(strict=True)
        self.label = label
        mf_path = self.path / "library_manifest.json"
        report_path = self.path / "audit_report.json"
        joint_path = self.path / "private_offline" / "selected_joint_records.jsonl"
        self.manifest = read_json(mf_path)
        self.audit = read_json(report_path)
        self.manifest_hash = sha_file(mf_path)
        self.joint_hash = sha_file(joint_path)
        mf = self.manifest
        if mf.get("schema") != "p1b_library_manifest_v1":
            raise InputError(f"{label}: not an original P1-B library manifest")
        self.dataset_id = mf.get("dataset_id")
        self.sample_sha = mf.get("training_sample_sha256")
        if not self.dataset_id or not self.sample_sha:
            raise InputError(f"{label}: dataset ID / training SHA absent")
        if mf.get("development_only") is not False:
            raise InputError(f"{label}: development-only source cannot be merged")
        if not self.audit.get("audit_passed"):
            raise InputError(f"{label}: source audit not passed")
        if self.audit.get("dataset_id") != self.dataset_id:
            raise InputError(f"{label}: audit dataset ID mismatch")
        if mf.get("roles") != list(ROLES) and set(mf.get("roles", ())) != set(ROLES):
            raise InputError(f"{label}: source roles mismatch")
        prov = mf.get("teacher_provenance")
        if not isinstance(prov, dict) or not prov.get("verified_mappo"):
            raise InputError(f"{label}: frozen MAPPO teacher provenance missing")
        self.teacher = prov
        self.joint = read_jsonl(joint_path)
        self.ids = [r.get("joint_id") for r in self.joint]
        if not self.joint or len(self.joint) != mf.get("n_experiences_per_role"):
            raise InputError(f"{label}: private joint record count mismatch")
        if len(self.ids) != len(set(self.ids)) or any(not x for x in self.ids):
            raise InputError(f"{label}: duplicate or blank joint ID")
        if self.audit.get("n_per_role") != len(self.ids):
            raise InputError(f"{label}: source audit count mismatch")
        self.roles = {}
        self.role_hashes = {}
        for role in ROLES:
            role_path = self.path / "role_libraries" / (role + ".jsonl")
            self.role_hashes[role] = sha_file(role_path)
            items = read_jsonl(role_path)
            if len(items) != len(self.joint):
                raise InputError(f"{label}: role length {role} differs from joint length")
            self.roles[role] = items
        for index, joint in enumerate(self.joint):
            ident = self.ids[index]
            if joint.get("dataset_id") != self.dataset_id:
                raise InputError(f"{label}: mixed-source private row {ident}")
            if joint.get("status") != "ok":
                raise InputError(f"{label}: non-ok private selected record {ident}")
            metrics = joint.get("metrics")
            if isinstance(metrics, dict):
                if metrics.get("cooperative_valid") is False or metrics.get("g_done") is False:
                    raise InputError(f"{label}: selected case fails cooperative filter {ident}")
            quality(joint)
            stratum(joint)
            for role in ROLES:
                r = self.roles[role][index]
                if set(r) != ROLE_KEYS:
                    raise InputError(f"{label}: role data field whitelist failure {role}")
                if r["schema"] != "p1b_local_case_v1" or r["role"] != role:
                    raise InputError(f"{label}: malformed role row for {role}")
                if r["experience_id"] != ident:
                    raise InputError(f"{label}: out-of-order role ID {role}:{ident}")
                finite_vector(r["observation_240"], 240, f"{role}.observation_240")
                n = 3 if role == "agent_g" else 4
                finite_vector(r["own_teacher_ratios"], n, f"{role}.own_teacher_ratios")
                finite_vector(r["own_teacher_env_action"], n, f"{role}.own_teacher_env_action")
                for a, b in zip(r["own_teacher_ratios"], r["own_teacher_env_action"]):
                    if abs(2 * a - 1 - b) > 2e-6:
                        raise InputError(f"{label}: role ratio/env mismatch {role}")
                    if a < -1e-6 or a > 1 + 1e-6:
                        raise InputError(f"{label}: out-of-range ratio {role}")
                if r["observation_240"] != joint["observations"][role]:
                    raise InputError(f"{label}: role observation/private mismatch {role}")
                if r["own_teacher_ratios"] != joint["teacher_actions_ratio"][role]:
                    raise InputError(f"{label}: role action/private mismatch {role}")
        self.fingerprints = [obs_signature(r) for r in self.joint]
        if len(set(self.fingerprints)) != len(self.fingerprints):
            raise InputError(f"{label}: exact duplicate joint observation detected within source")

    def describe(self) -> dict:
        return {
            "label": self.label,
            "library_dir_name": self.path.name,
            "dataset_id": self.dataset_id,
            "training_sample_sha256": self.sample_sha,
            "library_manifest_sha256": self.manifest_hash,
            "private_joint_sha256": self.joint_hash,
            "role_sha256": self.role_hashes,
            "n": len(self.joint),
            "teacher_checkpoint_sha256": self.teacher.get("checkpoint_sha256"),
        }


def validate_sources(old: SourceLibrary, new: SourceLibrary, holdout_paths: list[Path]):
    if old.dataset_id == new.dataset_id or old.sample_sha == new.sample_sha:
        raise InputError("Old and new sources refer to the same training dataset")
    if set(old.ids) & set(new.ids):
        ids = sorted(set(old.ids) & set(new.ids))
        raise InputError(f"Joint ID collision across sources: {ids[:3]}")
    duplicates = set(old.fingerprints) & set(new.fingerprints)
    if duplicates:
        raise InputError(f"Exact duplicate four-role observations across sources: {len(duplicates)}")
    for key in ("kind", "checkpoint_sha256"):
        a, b = old.teacher.get(key), new.teacher.get(key)
        if a is not None and b is not None and a != b:
            raise InputError(f"Teacher provenance {key} differs across sources")
    if not holdout_paths:
        raise InputError("Provide at least one --holdout-manifest for independent data audit")
    seen_holdouts = set()
    for path in holdout_paths:
        data = read_json(path)
        split = data.get("split")
        if split not in ("validation", "test"):
            raise InputError(f"Holdout manifest has invalid split: {path}")
        did = data.get("dataset_id")
        sha = data.get("sample_sha256")
        if not did or not sha:
            raise InputError(f"Holdout manifest missing dataset SHA/ID: {path}")
        if did in (old.dataset_id, new.dataset_id) or sha in (old.sample_sha, new.sample_sha):
            raise InputError(f"Train vs holdout dataset overlap: {path}")
        marker = (did, sha)
        if marker in seen_holdouts:
            raise InputError(f"Duplicate holdout supplied: {path}")
        seen_holdouts.add(marker)
    # A hash difference alone does not prove per-scene independence.
    return {"n_holdouts": len(seen_holdouts), "holdout_manifest_files":
            [{"path": str(p), "sha256": sha_file(p)} for p in holdout_paths]}


def ordered_balanced_new_indices(old: SourceLibrary, new: SourceLibrary) -> list[int]:
    old_count = collections.Counter(stratum(r) for r in old.joint)
    queues = collections.defaultdict(list)
    for idx, row in enumerate(new.joint):
        queues[stratum(row)].append(idx)
    for key in queues:
        queues[key].sort(key=lambda i: (-quality(new.joint[i]), i))
    # Count anchored old experiences, then prioritize under-covered strata.
    count = collections.Counter(old_count)
    result = []
    keys = sorted(queues)
    while len(result) != len(new.joint):
        options = [key for key in keys if queues[key]]
        if not options:
            raise InputError("Internal balanced-selection error")
        chosen = min(options, key=lambda key: (count[key], -len(queues[key]), key))
        result.append(queues[chosen].pop(0))
        count[chosen] += 1
    return result


def build_plan(old: SourceLibrary, new: SourceLibrary, sizes: tuple[int, ...]):
    max_total = len(old.joint) + len(new.joint)
    if any(x <= len(old.joint) or x > max_total for x in sizes):
        raise InputError(f"Target sizes must be in ({len(old.joint)}, {max_total}]")
    if list(sizes) != sorted(set(sizes)):
        raise InputError("Sizes must be strictly increasing and unique")
    ordered = {
        "standard": sorted(range(len(new.joint)), key=lambda i: (-quality(new.joint[i]), i)),
        "balanced": ordered_balanced_new_indices(old, new),
    }
    for method, seq in ordered.items():
        if len(seq) != len(new.joint) or len(set(seq)) != len(seq):
            raise InputError("Invalid retrieval selection order: " + method)
    return ordered


def make_manifest(old: SourceLibrary, new: SourceLibrary, base_id: str,
                  combined_sha: str, mode: str, size: int, report: dict,
                  source_descriptions: list[dict]) -> dict:
    meta = copy.deepcopy(old.manifest)
    meta["dataset_id"] = base_id
    meta["training_sample_sha256"] = combined_sha
    meta["training_sample_sha256_semantics"] = "sha256_of_two_source_dataset_provenances"
    meta["n_experiences_per_role"] = size
    meta["development_only"] = False
    meta["teacher_provenance"] = copy.deepcopy(old.teacher)
    meta["selection"] = {
        "method": "p2b_" + mode,
        "n_old_anchor": len(old.joint),
        "n_new_selected": size - len(old.joint),
        "rule": ("descending_p1b_selection_quality_score" if mode == "standard"
                 else "minimize_current_stratum_count_including_old_anchor_then_quality"),
    }
    meta["source_training_datasets"] = source_descriptions
    meta["source_dataset_sha256"] = [old.sample_sha, new.sample_sha]
    meta["p2b_selection_strata"] = report["strata_selected"]
    meta["p2b_id_namespace_version"] = "v2_stable_source_index"
    meta["p2b_id_namespace_rule"] = "composite_dataset_id::six_digit_global_source_index"
    meta["p2b_original_id_mapping"] = "private_offline/source_case_id_map.jsonl"
    meta["warnings"] = list(old.manifest.get("warnings", [])) + [
        "This is a composite of two independently collected TRAIN datasets.",
        "The training_sample_sha256 field is a provenance signature, not the SHA of one sample file.",
        "Original old 152 observations/actions and ordering are unchanged; IDs are remapped for LocalHistory.",
        "Original source dataset, joint ID and source scene ID are retained only in private_offline/source_case_id_map.jsonl.",
        "A successful p1b_audit and fresh p1d IR overlay are required before any evaluation.",
        "Holdout ID/SHA checks do not alone prove no identical individual scenes.",
        "No joint private observations or offline outcomes may enter online role prompts.",
    ]
    return meta


def make_report(old: SourceLibrary, new: SourceLibrary, indices: list[int],
                mode: str, size: int, provenance_sha: str) -> dict:
    candidates = old.joint + [new.joint[i] for i in indices]
    counter = collections.Counter(stratum(r) for r in candidates)
    vals = [quality(r) for r in candidates]
    return {
        "schema": "p2b_composite_selection_report_v1",
        "mode": mode,
        "capacity": size,
        "n_input": len(old.joint) + len(new.joint),
        "n_cooperative_candidates": len(old.joint) + len(new.joint),
        "n_selected": size,
        "n_old_anchor": len(old.joint),
        "n_new_selected": len(indices),
        "n_new_pool": len(new.joint),
        "n_after_relative_cost_filter": len(old.joint) + len(new.joint),
        "strata_selected": dict(sorted(counter.items())),
        "selection_quality_min": min(vals),
        "selection_quality_max": max(vals),
        "selection_quality_mean": sum(vals) / len(vals),
        "composite_provenance_sha256": provenance_sha,
        "hard_filter_rejections": "Already applied by each original P1-B source selector; not re-run",
        "limitations": [
            "Standard ranks the already-selected new source by its P1-B quality score; it is not a fresh two-pool P1-B optimization.",
            "Balanced uses original P1-B strata and case-level selection quality scores.",
            "The two source selectors may have different quality-score normalization.",
            "No Qwen inference or experiment performance has been measured.",
        ],
    }


def build_one(old: SourceLibrary, new: SourceLibrary, indices: list[int],
              mode: str, size: int, output_path: Path, base_id: str,
              provenance_sha: str, source_descriptions: list[dict]) -> dict:
    """Build version-2 library using dataset-prefixed, stable unique IDs.

    Global numeric indices are assigned once from BOTH complete input sources,
    not from the selected subset: old row 0..n_old-1, new n_old..n_old+n_new-1.
    Thus a case has exactly the same identity in all six output libraries.
    """
    if output_path.exists():
        raise FileExistsError(f"Refuse to overwrite existing directory: {output_path}")
    if not isinstance(base_id, str) or not base_id:
        raise InputError("Blank composite dataset ID")
    if len(set(indices)) != len(indices) or any(i < 0 or i >= len(new.joint) for i in indices):
        raise InputError("Invalid new-source selection indices")
    if size != len(old.joint) + len(indices):
        raise InputError("Target size mismatch")

    inputs = [(old, i, i) for i in range(len(old.joint))]
    inputs += [(new, i, len(old.joint) + i) for i in indices]
    selected_joint = []
    role_tables = {role: [] for role in ROLES}
    origin_map = []
    for source, index, global_index in inputs:
        original_joint = source.joint[index]
        original_case_id = original_joint["joint_id"]
        if not isinstance(original_case_id, str) or not original_case_id:
            raise InputError("Missing original source case ID")
        composite_id = f"{base_id}::{global_index:06d}"
        joint = copy.deepcopy(original_joint)
        joint["dataset_id"] = base_id
        joint["joint_id"] = composite_id
        selected_joint.append(joint)
        for role in ROLES:
            local = copy.deepcopy(source.roles[role][index])
            if local.get("experience_id") != original_case_id:
                raise InputError("Unmatched source and role ID before rekeying")
            local["experience_id"] = composite_id
            role_tables[role].append(local)
        origin_map.append({
            "composite_joint_id": composite_id,
            "original_dataset_id": source.dataset_id,
            "original_training_sample_sha256": source.sample_sha,
            "original_joint_id": original_case_id,
            "original_scene_id": original_joint.get("scene_id"),
            "original_selected_source_index": index,
        })

    ids = [r["joint_id"] for r in selected_joint]
    if len(ids) != size or len(set(ids)) != size:
        raise InputError("Generated duplicate or missing composite IDs")
    if any(not key.startswith(base_id + "::") for key in ids):
        raise InputError("Bad generated composite ID prefix")
    for role in ROLES:
        if [r["experience_id"] for r in role_tables[role]] != ids:
            raise InputError("Role/joint ID misalignment after rekeying")

    report = make_report(old, new, indices, mode, size, provenance_sha)
    report["id_namespace"] = "composite_dataset_id::stable_source_index"
    manifest = make_manifest(old, new, base_id, provenance_sha, mode, size,
                             report, source_descriptions)
    parent = output_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".p2b_v2_staging_", dir=str(parent)))
    try:
        (tmp / "role_libraries").mkdir()
        (tmp / "private_offline").mkdir()
        emit_jsonl(tmp / "private_offline" / "selected_joint_records.jsonl", selected_joint)
        emit_jsonl(tmp / "private_offline" / "source_case_id_map.jsonl", origin_map)
        index_data = [{"joint_id": rec["joint_id"],
                       "stratum": rec["selection_stratum"],
                       "quality_score": quality(rec)} for rec in selected_joint]
        emit_json(tmp / "private_offline" / "selection_index.json", index_data)
        for role in ROLES:
            emit_jsonl(tmp / "role_libraries" / (role + ".jsonl"), role_tables[role])
        emit_json(tmp / "library_manifest.json", manifest)
        emit_json(tmp / "selection_report.json", report)
        (tmp / "selection_report.txt").write_text(
            f"P2-B composite library v2: {mode}\nSize: {size}\n"
            f"Old anchor: {len(old.joint)}\nNew examples: {len(indices)}\n"
            f"Original datasets: {old.dataset_id} + {new.dataset_id}\n"
            "Case IDs are rekeyed to the composite dataset namespace.\n"
            "Original IDs are in private_offline/source_case_id_map.jsonl.\n"
            "Pass p1b_audit.py, LocalHistory, and IRHistory before evaluation.\n",
            encoding="utf-8",
        )
        if output_path.exists():
            raise FileExistsError(f"Refuse existing output: {output_path}")
        tmp.rename(output_path)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    return {"directory": str(output_path), "mode": mode, "size": size,
            "n_old": len(old.joint), "n_new": len(indices),
            "strata": report["strata_selected"],
            "mapping_sha256": sha_file(output_path / "private_offline" / "source_case_id_map.jsonl")}


def check_no_existing(output: Path, sizes: tuple[int, ...]):
    targets = [output / f"p2b_{mode}_{size}_v2"
               for mode in ("standard", "balanced") for size in sizes]
    if any(path.exists() for path in targets):
        existing = [str(path) for path in targets if path.exists()]
        raise FileExistsError("Refusing to overwrite: " + ", ".join(existing))
    return targets


def run(args) -> dict:
    old = SourceLibrary(Path(args.old_library), "old")
    new = SourceLibrary(Path(args.new_library), "new")
    holdouts = [Path(p).resolve(strict=True) for p in args.holdout_manifest]
    holdout_report = validate_sources(old, new, holdouts)
    if args.expect_old and len(old.joint) != args.expect_old:
        raise InputError(f"Old source count changed: expected {args.expect_old}")
    if args.expect_new and len(new.joint) != args.expect_new:
        raise InputError(f"New source count changed: expected {args.expect_new}")
    sizes = tuple(int(s.strip()) for s in args.sizes.split(","))
    orders = build_plan(old, new, sizes)
    output = Path(args.output_dir).resolve()
    if output == old.path or output == new.path:
        raise InputError("Output must not equal either source library")
    try:
        output.relative_to(old.path)
    except ValueError:
        pass
    else:
        raise InputError("Cannot write inside old source library")
    try:
        output.relative_to(new.path)
    except ValueError:
        pass
    else:
        raise InputError("Cannot write inside new source library")
    targets = check_no_existing(output, sizes) if args.execute else [
        output / f"p2b_{mode}_{size}_v2"
        for mode in ("standard", "balanced") for size in sizes]
    if args.execute and (output / "p2b_composite_build_summary_v2.json").exists():
        raise FileExistsError("Refusing to overwrite P2-B build summary")
    source_descriptions = [old.describe(), new.describe()]
    source_marker = [{"dataset_id": x["dataset_id"],
                      "sample_sha256": x["training_sample_sha256"]}
                     for x in source_descriptions]
    signature = sha_bytes(canonical_json(source_marker).encode("utf-8"))
    merged_id = "p2b_composite_train_" + signature[:12]
    summary = {
        "schema": "p2b_builder_summary_v2", "executed": bool(args.execute),
        "old_count": len(old.joint), "new_count": len(new.joint),
        "pool_count": len(old.joint) + len(new.joint),
        "holdout_validation": holdout_report,
        "composite_dataset_id": merged_id,
        "composite_provenance_sha256": signature,
        "source_libraries": source_descriptions,
        "outputs": [],
    }
    for mode in ("standard", "balanced"):
        for size in sizes:
            n_new = size - len(old.joint)
            target = output / f"p2b_{mode}_{size}_v2"
            indices = orders[mode][:n_new]
            if args.execute:
                row = build_one(old, new, indices, mode, size, target,
                                merged_id, signature, source_descriptions)
            else:
                counter = collections.Counter(
                    stratum(x) for x in old.joint + [new.joint[i] for i in indices])
                row = {"directory": str(target), "mode": mode,
                       "size": size, "n_old": len(old.joint),
                       "n_new": n_new, "strata": dict(sorted(counter.items()))}
            summary["outputs"].append(row)
    if args.execute:
        summary_path = output / "p2b_composite_build_summary_v2.json"
        if summary_path.exists():
            raise FileExistsError(f"Summary already exists: {summary_path}")
        emit_json(summary_path, summary)
    return summary


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-library", default=DEFAULT_OLD)
    p.add_argument("--new-library", default=DEFAULT_NEW)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    p.add_argument("--holdout-manifest", action="append", required=True)
    p.add_argument("--sizes", default="300,500,1000")
    p.add_argument("--expect-old", type=int, default=152)
    p.add_argument("--expect-new", type=int, default=1209)
    p.add_argument("--execute", action="store_true",
                   help="Write new P2-B libraries; without it only validate and show plan")
    return p.parse_args()


if __name__ == "__main__":
    try:
        report = run(cli())
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("STATUS: new libraries written; run p1b_audit and IR build before scoring."
          if report["executed"] else
          "STATUS: read-only preflight passed; add --execute to write new libraries.")
