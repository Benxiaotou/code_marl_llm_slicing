"""40-scene Qwen node-calibration pilot with a fixed 25% action blend."""
from __future__ import annotations
import contextlib, datetime as dt, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "p2f_analysis"))
import qwen_full200 as full
import val5601_counterfactual_replay as replay

SCENES = list(range(200))
MODEL = full.MODEL
ROLES = replay.ROLES
K_ROLES = replay.K_ROLES
ALPHA = 0.25
PROMPT_PATH = Path(__file__).with_name("node_calibration_prompt.txt")

def sha_text(s): return hashlib.sha256(s.encode("utf-8")).hexdigest()

def make_policy_class(library, ir, cached, client, prompt, candidate):
    from p2a.policy import IRRiskRerankPolicy
    class Policy(IRRiskRerankPolicy):
        def __init__(self):
            self.candidate = candidate
            super().__init__(library, ir, client=client, examples=3, seed=1)
        def _make_messages(self, role, observation, matches):
            original = super()._make_messages(role, observation, matches)
            expected = cached[self.scene_id]["per_role"][role]["messages"]
            if original != expected:
                raise RuntimeError("base prompt mismatch")
            if self.candidate and role in K_ROLES:
                original = [dict(m) for m in original]
                original[0]["content"] += "\n" + prompt
            return original
    return Policy

def score_actions(env, actions, scene):
    from p1a_runner import load_scene_local_observations, _get_step_metrics
    load_scene_local_observations(env, scene)
    import random
    random.seed(1 + scene); np.random.seed(1 + scene)
    before = len(env.g_done_list)
    arr = {r: np.asarray(v, dtype=np.float32) for r,v in actions.items()}
    _, rewards, _, _, _ = env.step(arr)
    return _get_step_metrics(env, before, rewards)

def main():
    prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    manifest_path = ROOT / "p1b_data/p2b_val_seed5601.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    from p1c.retrieval import LocalHistory
    from p1d.ir import IRHistory
    from p2a.rerank import RerankConfig
    from p2e_prompt.val5601_eval import verify_environment
    import p1a_runner as base, p1c_runner
    main_module, _ = base._config(ROOT, ROOT / "configs_main.yaml")
    base._compat_pickle_main_classes(main_module)
    verify_environment(ROOT, manifest)
    library = LocalHistory(ROOT / "p2b_libraries/p2b_standard_1000_v2")
    ir = IRHistory(ROOT / "p2b_ir_libraries/p2b_standard_1000_ir_v2", library)
    qfolder = ROOT / "p2b_runs/val5601_standard1000_p2a200"
    cached = replay.read_jsonl(qfolder / "records.jsonl")
    out = ROOT / "p2g_autonomous" / "results" / ("node_full_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    out.mkdir(parents=True, exist_ok=False); runtime = out / "env_runtime"; runtime.mkdir()
    with (out / "env_stdout.log").open("x", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log): env = p1c_runner.create_eval_env(ROOT, manifest, "configs_main.yaml")
    env.path_reward_save = str(runtime); env.path_env_save = str(runtime)
    with (out / "requests.jsonl").open("x", encoding="utf-8") as ledger:
        client = full.AuditedClient(
            __import__("p1a.policy", fromlist=["ChatCompletionsClient"]).ChatCompletionsClient(
                base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"], model=MODEL,
                temperature=.1, max_tokens=4096, read_timeout=120, stream=True), ledger)
        # Preflight reconstructs all 160 prompts before the first request.
        for scene in SCENES:
            for candidate in (False, True):
                P = make_policy_class(library, ir, cached, client, prompt, candidate)
                probe = P(); probe.set_scene(scene)
                from p1a_runner import load_scene_local_observations
                obs = load_scene_local_observations(env, scene)
                for role in ROLES:
                    matches = probe.library.query(role, obs[role], top_k=3)
                    probe._make_messages(role, obs[role], matches)
        summaries = {"original": [], "candidate_node": [], "blended_25": []}
        records = []
        for scene in SCENES:
            scene_actions = {}
            for label, candidate in (("original", False), ("candidate_node", True)):
                P = make_policy_class(library, ir, cached, client, prompt, candidate)
                client.arm = label; client.scene = scene
                policy = P(); policy.set_scene(scene)
                from p1a_runner import load_scene_local_observations
                obs = load_scene_local_observations(env, scene)
                actions, entries = policy.act(obs)
                scene_actions[label] = {r: list(np.asarray(v, dtype=float)) for r,v in actions.items()}
                metrics = score_actions(env, scene_actions[label], scene)
                summaries[label].append(metrics)
                records.append({"scene":scene,"arm":label,"metrics":metrics,"env_actions":scene_actions[label],"per_role":entries})
            blended = {r:list(scene_actions["original"][r]) for r in ROLES}
            for role in K_ROLES:
                blended[role][:2] = (np.asarray(scene_actions["original"][role][:2]) +
                    ALPHA*(np.asarray(scene_actions["candidate_node"][role][:2]) -
                           np.asarray(scene_actions["original"][role][:2]))).tolist()
            blended_metrics = score_actions(env, blended, scene)
            summaries["blended_25"].append(blended_metrics)
            records.append({"scene":scene,"arm":"blended_25","metrics":blended_metrics,"env_actions":blended})
        for name, rows in summaries.items():
            summaries[name] = replay.aggregate([{"metrics":r} for r in rows])
        for row in records:
            row["status"]="ok"
        with (out / "records.jsonl").open("x", encoding="utf-8") as f:
            for row in records: f.write(json.dumps(row, allow_nan=False) + "\n")
        result = {"status":"COMPLETE","scenes":SCENES,"alpha":ALPHA,"model":MODEL,
                  "prompt_sha256":sha_text(prompt),"summaries":summaries,
                  "successful_calls":client.count,"api_cost_usd_estimate":0.0,
                  "limitations":"development expansion on the same validation pool as prior work; one draw per arm; blended actions are an offline deterministic post-processing arm"}
        (out/"summary.json").write_text(json.dumps(result, indent=2)+"\n",encoding="utf-8")
        print(json.dumps({"output":str(out),"summaries":summaries,"successful_calls":client.count},separators=(",",":")),flush=True)

if __name__ == "__main__":
    main()
