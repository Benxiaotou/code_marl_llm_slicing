"""Compare P1-A records with the pre-existing P0 MAPPO records on exact scene IDs.

No model calls. This compares observed outcomes, and DOES NOT guarantee
identical environment versions or stochastic seeds without additional auditing.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


FIELDS = ("V", "V_g", "cost_sum")


def _bool(x):
    if isinstance(x, bool):
        return x
    if x in (0, 1):
        return bool(x)
    raise ValueError("Not a Boolean outcome: {!r}".format(x))


def _normalize(record, role):
    if role == "p1a":
        if record.get("status") != "ok":
            raise ValueError("P1-A contains failed/unscored decisions")
        scene = record["original_scene_id"]
        data = record["metrics"]
        if record.get("fallback_used", False):
            raise ValueError("Fallback contaminates P1-A records")
    else:
        if record.get("used_fallback", False) or record.get("decision_error"):
            raise ValueError("MAPPO reference has fallback or errors")
        scene = record.get("original_scene_index", record.get("original_scene_id"))
        if scene is None:
            raise ValueError("MAPPO record lacks original scene ID")
        data = record.get("metrics", record)
    g_done = _bool(data["g_done"])
    ir = all(float(data["V_" + a]) >= float(data["V_" + a + "_0"]) - 1e-8
             for a in ("k1", "k2", "k3")) and float(data["V_g"]) >= -1e-8
    ret = {"scene_id": int(scene), "g_done": g_done, "ir": ir,
           "cooperative_valid": g_done and ir}
    for key in FIELDS:
        ret[key] = float(data[key])
        if not np.isfinite(ret[key]):
            raise ValueError("Invalid metric " + key)
    return ret


def load_records(path, role):
    result = {}
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            obj = _normalize(json.loads(line), role)
            scene = obj["scene_id"]
            if scene in result:
                raise ValueError("Duplicate scene {} in {}".format(scene, path))
            result[scene] = obj
    return result


def compare(p1a_file, mappo_file):
    p1 = load_records(p1a_file, "p1a")
    p0 = load_records(mappo_file, "mappo")
    if not p1 or set(p1) != set(p0):
        raise ValueError("Scene IDs differ or are empty. P1-only: {}; MAPPO-only: {}".format(
            sorted(set(p1) - set(p0)), sorted(set(p0) - set(p1))))
    scene_rows = []
    for scene_id in sorted(p1):
        a, b = p1[scene_id], p0[scene_id]
        row = {"scene_id": scene_id,
               "qwen_admission": int(a["g_done"]), "mappo_admission": int(b["g_done"]),
               "delta_admission": int(a["g_done"]) - int(b["g_done"]),
               "qwen_cooperative": int(a["cooperative_valid"]),
               "mappo_cooperative": int(b["cooperative_valid"]),
               "delta_cooperative": int(a["cooperative_valid"]) - int(b["cooperative_valid"])}
        for key in FIELDS:
            row["qwen_" + key] = a[key]
            row["mappo_" + key] = b[key]
            row["delta_" + key] = a[key] - b[key]
        scene_rows.append(row)
    summary = {
        "n_paired": len(scene_rows),
        "original_scene_ids": sorted(p1),
        "warning": "Descriptive paired comparison only. Validate identical main.py, metrics, samples and stochastic seeds.",
    }
    for key in ("admission", "cooperative", "V", "V_g", "cost_sum"):
        summary["qwen_" + key] = float(np.mean([r["qwen_" + key] for r in scene_rows]))
        summary["mappo_" + key] = float(np.mean([r["mappo_" + key] for r in scene_rows]))
        summary["paired_delta_" + key] = float(np.mean([r["delta_" + key] for r in scene_rows]))
    return summary, scene_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--p1a", required=True, help="P1-A records.jsonl")
    p.add_argument("--mappo", required=True, help="P0 MAPPO mappo_records.jsonl")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    summary, rows = compare(args.p1a, args.mappo)
    (out / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    with (out / "paired_scenes.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
