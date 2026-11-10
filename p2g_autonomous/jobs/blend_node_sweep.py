"""Offline conservative interpolation of fresh candidate node actions."""
from __future__ import annotations
import contextlib, datetime as dt, json
from pathlib import Path
import numpy as np
from cross_action_blocks import ROOT, SOURCE, read_rows, score, aggregate

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
ROLES = ("agent_k1", "agent_k2", "agent_k3")

def main():
    original = read_rows(SOURCE / "full_original_records.jsonl")
    candidate = read_rows(SOURCE / "full_link_calibration_records.jsonl")
    manifest_path = ROOT / "p1b_data/p2b_val_seed5601.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    import sys
    sys.path.insert(0, str(ROOT))
    import p1a_runner
    import p2e_prompt.val5601_eval as adapter
    main_module, _ = p1a_runner._config(ROOT, ROOT / "configs_main.yaml")
    p1a_runner._compat_pickle_main_classes(main_module)
    adapter.verify_environment(ROOT, manifest)
    import p1c_runner
    out = ROOT / "p2g_autonomous" / "results" / ("blend_node_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    out.mkdir(parents=True, exist_ok=False)
    runtime = out / "env_runtime"; runtime.mkdir()
    with (out / "env_stdout.log").open("x", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log):
            env = p1c_runner.create_eval_env(ROOT, manifest, "configs_main.yaml")
    env.path_reward_save = str(runtime); env.path_env_save = str(runtime)
    summaries = {}
    transitions = {}
    with (out / "records.jsonl").open("x", encoding="utf-8") as sink:
        for alpha in ALPHAS:
            name = "alpha_" + str(alpha).replace(".", "p")
            rows = []; trans = {"original_coop_to_blend_loss": 0, "original_coop_to_blend_win": 0}
            for scene in range(200):
                actions = {}
                for role in ROLES:
                    ov = np.asarray(original[scene]["env_actions"][role], dtype=float)
                    cv = np.asarray(candidate[scene]["env_actions"][role], dtype=float)
                    vec = ov.copy(); vec[:2] = ov[:2] + alpha * (cv[:2] - ov[:2])
                    actions[role] = vec.tolist()
                actions["agent_g"] = list(original[scene]["env_actions"]["agent_g"])
                metrics = score(env, actions, scene)
                rows.append(metrics)
                om = bool(original[scene]["metrics"]["cooperative_valid"])
                bm = bool(metrics["cooperative_valid"])
                if om and not bm: trans["original_coop_to_blend_loss"] += 1
                if not om and bm: trans["original_coop_to_blend_win"] += 1
                sink.write(json.dumps({"alpha": alpha, "scene": scene, "metrics": metrics}, allow_nan=False) + "\n")
            summaries[name] = aggregate(rows); transitions[name] = trans
    payload = {"status":"COMPLETE","alphas":list(ALPHAS),"summaries":summaries,"transitions":transitions,"api_calls":0,"api_cost_usd_estimate":0.0}
    (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output":str(out),"summaries":summaries,"transitions":transitions}, separators=(",",":")), flush=True)

if __name__ == "__main__":
    main()
