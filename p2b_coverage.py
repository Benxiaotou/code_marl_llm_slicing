#!/usr/bin/env python3
"""Audit retrieval coverage from P1-C / P2-A per-scene run records.

No model calls or environment runs. This reads per-role audit distances only,
never prompts, current cross-agent observations, or historical private data.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
import zipfile
from pathlib import Path

ROLES = ("agent_k1", "agent_k2", "agent_k3", "agent_g")


def load_run(path):
    path = Path(path)
    if path.is_dir():
        matches = list(path.rglob("records.jsonl"))
        if not (path / "records.jsonl").exists() and len(matches) != 1:
            raise ValueError("Expected a run directory containing exactly one records.jsonl")
        records_file = path / "records.jsonl" if (path / "records.jsonl").exists() else matches[0]
        manifest_file = records_file.with_name("manifest.json")
        if not manifest_file.is_file():
            raise ValueError("Run is missing manifest.json")
        record_bytes = records_file.read_bytes()
        manifest_bytes = manifest_file.read_bytes()
    elif path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            records = [n for n in zf.namelist() if n.endswith("records.jsonl")]
            if len(records) != 1:
                raise ValueError("ZIP must contain exactly one records.jsonl (extract first for multi-run archives)")
            name = records[0]
            record_bytes = zf.read(name)
            manifest_bytes = zf.read(name.rsplit("/", 1)[0] + "/manifest.json" if "/" in name else "manifest.json")
    else:
        raise ValueError("Run path must be a run folder or ZIP archive")

    manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
    rows = []
    for line_num, line in enumerate(record_bytes.decode("utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("status") != "ok" or obj.get("fallback_used"):
            raise ValueError("Invalid/fallback scene at line {}".format(line_num))
        scene_id = obj.get("original_scene_id")
        if scene_id is None:
            raise ValueError("Missing scene ID at line {}".format(line_num))
        met = obj.get("metrics") or {}
        if not isinstance(met.get("cooperative_valid"), bool):
            raise ValueError("Missing boolean cooperative_valid at line {}".format(line_num))
        row = {"scene_id": str(scene_id), "cooperative_valid": met["cooperative_valid"],
               "g_done": bool(met["g_done"]), "V": float(met["V"]),
               "cost_sum": float(met["cost_sum"])}
        for role in ROLES:
            details = (obj.get("per_role") or {}).get(role) or {}
            ds = details.get("distances_for_audit_only")
            if not isinstance(ds, list) or len(ds) < 1:
                raise ValueError("Missing per-role retrieval distance for {} at line {}".format(role, line_num))
            d = float(ds[0])
            if not math.isfinite(d) or d < 0:
                raise ValueError("Invalid nearest neighbor distance for {} at line {}".format(role, line_num))
            row[role] = d
        if any(not math.isfinite(row[k]) for k in ("V", "cost_sum")):
            raise ValueError("Nonfinite V / cost at line {}".format(line_num))
        rows.append(row)
    ids = [r["scene_id"] for r in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Empty run or duplicate scene IDs")
    expected = manifest.get("scene_ids")
    if isinstance(expected, list) and (len(expected) != len(rows) or {str(i) for i in expected} != set(ids)):
        raise ValueError("manifest scene_ids mismatch")
    return manifest, rows, hashlib.sha256(record_bytes).hexdigest()


def pct(values, q):
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values)-1) * q
    i = int(pos)
    return values[i] + (values[min(i+1, len(values)-1)] - values[i]) * (pos-i)


def summarize(manifest, rows, sha):
    s = {"schema": "p2b_retrieval_coverage_v1", "run_records_sha256": sha,
         "method": manifest.get("method"), "eval_dataset_id": manifest.get("eval_dataset_id"),
         "eval_dataset_sha256": manifest.get("eval_dataset_sha256"),
         "library_manifest_sha256": manifest.get("library_manifest_sha256"),
         "n": len(rows), "cooperative_valid_count": sum(r["cooperative_valid"] for r in rows),
         "cooperative_valid_rate": statistics.mean(r["cooperative_valid"] for r in rows),
         "V_mean": statistics.mean(r["V"] for r in rows),
         "cost_mean": statistics.mean(r["cost_sum"] for r in rows),
         "roles": {}}
    for role in ROLES:
        ordered = sorted(rows, key=lambda r: (r[role], r["scene_id"]))
        distances = [r[role] for r in rows]
        qn = max(1, len(rows) // 4)
        s["roles"][role] = {
            "nearest_mean": statistics.mean(distances),
            "nearest_p50": pct(distances, 0.5),
            "nearest_p90": pct(distances, 0.9),
            "nearest_max": max(distances),
            "nearest_over_0_35_count": sum(d > 0.35 for d in distances),
            "bottom_distance_quartile_coop_rate": statistics.mean(r["cooperative_valid"] for r in ordered[:qn]),
            "top_distance_quartile_coop_rate": statistics.mean(r["cooperative_valid"] for r in ordered[-qn:])
        }
    return s


def compare(ma, ra, mb, rb):
    a_id = ma.get("eval_dataset_id")
    b_id = mb.get("eval_dataset_id")
    a_sha = ma.get("eval_dataset_sha256")
    b_sha = mb.get("eval_dataset_sha256")
    if not a_id or not a_sha or a_id != b_id or a_sha != b_sha:
        raise ValueError("Cannot compare: evaluation dataset identity or SHA256 differ/missing")
    a = {r["scene_id"]: r for r in ra}
    b = {r["scene_id"]: r for r in rb}
    if set(a) != set(b):
        raise ValueError("Cannot compare: scene IDs differ")
    ids = sorted(a)
    out = {"n_paired": len(ids), "dataset_id": a_id,
           "cooperative_valid_rate_delta": statistics.mean(b[i]["cooperative_valid"] - a[i]["cooperative_valid"] for i in ids),
           "mean_V_delta": statistics.mean(b[i]["V"] - a[i]["V"] for i in ids),
           "mean_cost_delta": statistics.mean(b[i]["cost_sum"] - a[i]["cost_sum"] for i in ids),
           "invalid_to_valid": sum((not a[i]["cooperative_valid"]) and b[i]["cooperative_valid"] for i in ids),
           "valid_to_invalid": sum(a[i]["cooperative_valid"] and (not b[i]["cooperative_valid"]) for i in ids),
           "roles": {}}
    for role in ROLES:
        out["roles"][role] = {
            "paired_nearest_mean_delta": statistics.mean(b[i][role] - a[i][role] for i in ids),
            "nearest_p50_delta": pct([b[i][role] for i in ids], .5) - pct([a[i][role] for i in ids], .5),
            "nearest_p90_delta": pct([b[i][role] for i in ids], .9) - pct([a[i][role] for i in ids], .9)
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Run directory or ZIP with records.jsonl + manifest.json")
    parser.add_argument("--compare", help="Optional second run on exactly the same eval dataset")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    ma, ra, sha = load_run(args.run)
    result = {"baseline": summarize(ma, ra, sha),
              "note": "Descriptive retrieval distances only; association is not causation. Compare distance scales only if feature scaling/normalization is fixed."}
    if args.compare:
        mb, rb, shb = load_run(args.compare)
        result["candidate"] = summarize(mb, rb, shb)
        result["paired_comparison"] = compare(ma, ra, mb, rb)
    dest = Path(args.output_dir)
    if dest.exists() and any(dest.iterdir()):
        raise ValueError("Refusing to overwrite a nonempty output directory")
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "coverage_summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with (dest / "coverage_per_scene.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["scene_id", "cooperative_valid", "g_done", "V", "cost_sum", *ROLES])
        writer.writeheader()
        writer.writerows(sorted(ra, key=lambda r: r["scene_id"]))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
