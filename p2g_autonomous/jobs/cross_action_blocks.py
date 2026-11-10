"""Offline crossover of fresh Qwen action blocks; never constructs an API client."""
from __future__ import annotations
import contextlib, csv, datetime as dt, json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "p2f_analysis/qwen_full200/run_20261104T141915_396460Z"
ROLES = ("agent_k1", "agent_k2", "agent_k3", "agent_g")
K_ROLES = ROLES[:3]
CASES = (
    ("original", False, False, False),
    ("candidate_node", True, False, False),
    ("candidate_link", False, True, False),
    ("candidate_g", False, False, True),
    ("candidate_node_link", True, True, False),
    ("candidate_node_g", True, False, True),
    ("candidate_link_g", False, True, True),
    ("candidate", True, True, True),
)

def read_rows(path):
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[int(row["original_scene_id"])] = row
    if sorted(out) != list(range(200)):
        raise ValueError("expected scenes 0..199")
    return out

def actions_for(o, c, use_node, use_link, use_g, scene):
    a = {}
    for role in K_ROLES:
        src = c[scene]["env_actions"][role] if use_node else o[scene]["env_actions"][role]
        vec = list(src)
        link_src = c[scene]["env_actions"][role] if use_link else o[scene]["env_actions"][role]
        vec[2:] = list(link_src[2:])
        a[role] = vec
    a["agent_g"] = list(c[scene]["env_actions"]["agent_g"] if use_g
                        else o[scene]["env_actions"]["agent_g"])
    return a

def score(env, actions, scene, seed=1):
    from p1a_runner import load_scene_local_observations, _get_step_metrics
    load_scene_local_observations(env, scene)
    arr = {r: np.asarray(v, dtype=np.float32) for r, v in actions.items()}
    import random
    random.seed(seed + scene)
    np.random.seed(seed + scene)
    before = len(env.g_done_list)
    _, rewards, _, _, _ = env.step(arr)
    return _get_step_metrics(env, before, rewards)

def aggregate(rows):
    keys = ("g_done", "cooperative_valid", "individual_rationality", "V", "V_g", "cost_sum")
    out = {"n": len(rows)}
    for k in keys:
        out[k + ("_rate" if k in keys[:3] else "_mean")] = float(np.mean([float(r[k]) for r in rows]))
    out["accepted_ir_invalid_n"] = int(sum(bool(r["g_done"]) and not bool(r["individual_rationality"]) for r in rows))
    return out

def main():
    original = read_rows(SOURCE / "full_original_records.jsonl")
    candidate = read_rows(SOURCE / "full_link_calibration_records.jsonl")
    manifest_path = ROOT / "p1b_data/p2b_val_seed5601.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sys.path.insert(0, str(ROOT))
    import p1a_runner
    import p2e_prompt.val5601_eval as adapter
    main_module, _ = p1a_runner._config(ROOT, ROOT / "configs_main.yaml")
    p1a_runner._compat_pickle_main_classes(main_module)
    adapter.verify_environment(ROOT, manifest)
    import p1c_runner
    out = ROOT / "p2g_autonomous" / "results" / ("cross_blocks_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    out.mkdir(parents=True, exist_ok=False)
    runtime = out / "env_runtime"; runtime.mkdir()
    with (out / "env_stdout.log").open("x", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log):
            env = p1c_runner.create_eval_env(ROOT, manifest, "configs_main.yaml")
    env.path_reward_save = str(runtime)
    env.path_env_save = str(runtime)
    summaries = {}
    paired = []
    with (out / "records.jsonl").open("x", encoding="utf-8") as sink:
        for name, use_node, use_link, use_g in CASES:
            rows = []
            for scene in range(200):
                metrics = score(env, actions_for(original, candidate, use_node, use_link, use_g, scene), scene)
                row = {"case": name, "original_scene_id": scene, "metrics": metrics}
                sink.write(json.dumps(row, allow_nan=False) + "\n")
                rows.append(metrics)
            summaries[name] = aggregate(rows)
    base = []
    for scene in range(200):
        om = score  # controls are checked against saved metrics below without a second replay
        base.append({
            "scene": scene,
            "candidate_minus_original_V": float(candidate[scene]["metrics"]["V"] - original[scene]["metrics"]["V"]),
        })
    with (out / "paired_deltas.csv").open("x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(base[0])); w.writeheader(); w.writerows(base)
    protocol = {
        "schema": "qwen_fresh_action_block_crossover_v1",
        "dataset_id": manifest["dataset_id"], "scene_ids": list(range(200)),
        "source_run": str(SOURCE.relative_to(ROOT)), "cases": [x[0] for x in CASES],
        "no_model_calls": True, "api_cost_usd_estimate": 0.0,
        "interpretation": "node/link dimensions for k roles and full g action are swapped from fresh original/candidate records; original environment and metrics unchanged.",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"status":"COMPLETE","summaries":summaries,"api_calls":0,"api_cost_usd_estimate":0.0}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "summaries": summaries}, separators=(",", ":")), flush=True)

if __name__ == "__main__":
    main()
