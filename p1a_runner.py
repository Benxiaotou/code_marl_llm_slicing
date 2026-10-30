"""P1-A GEANT four-role decentralized Qwen experiment (no MARL/Guard in policy).

This runner is deliberately independent of P0 runner / P0 policy. It imports
original main.CustomMultiAgentEnv only to evaluate a completed joint action.
"""
import argparse
import copy
import csv
import datetime
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from p1a.policy import ChatCompletionsClient, DecentralizedPolicy, MockClient, PolicyFailure
from p1a.local_obs import AGENTS


METRICS = (
    "V", "V_k1", "V_k2", "V_k3", "V_g", "V_k1_0", "V_k2_0", "V_k3_0",
    "X_k1", "X_k2", "X_k3", "cost_k1", "cost_k2", "cost_k3",
    "revenue_k1", "revenue_k2", "revenue_k3", "revenue_g",
)


def get_args(argv=None):
    p = argparse.ArgumentParser(description="P1-A: independent local Qwen agents")
    p.add_argument("--dataset", choices=["mini", "full"], default="mini")
    p.add_argument("--mini-file", default="P0_small_samples.pkl")
    p.add_argument("--mini-map", default="P0_small_samples.indices.json")
    p.add_argument("--config", default="configs_main.yaml")
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--start", type=int, default=0, help="Start offset in dataset order")
    p.add_argument("--scenes", default="", help="Comma-separated ORIGINAL scene IDs (e.g. 0,1,2,299,599)")
    p.add_argument("--policy", choices=["qwen", "mock"], default="qwen")
    p.add_argument("--model", default=os.getenv("P1A_MODEL", "qwen3-next-80b-a3b-instruct"))
    p.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "http://192.168.131.35:30001/v1"))
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--api-timeout", type=float, default=120.0,
                   help="Per-read timeout, NOT a total request wall-clock deadline")
    p.add_argument("--no-stream", action="store_true")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--on-error", choices=["abort", "record"], default="abort")
    p.add_argument("--output-dir", default="")
    return p.parse_args(argv)


def _json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding="utf-8")


def _hash_file(path):
    if not Path(path).is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for piece in iter(lambda: f.read(1024 * 1024), b""):
            h.update(piece)
    return h.hexdigest()


def _config(root, file_name):
    # main.py imports XuanCe on module import; it is NOT run as __main__ here.
    import main
    conf = dict(main.get_configs(file_dir=str(file_name)))
    conf["root"] = str(root)
    conf["data_root"] = str(root / "data")
    conf["result_root"] = str(root / "result")
    conf["path_reward_save"] = str(root / "result" / "reward_save")
    conf["path_env_save"] = str(root / "result" / "env_save")
    conf["save_attach_word"] = str(conf["max_num_steps"]) + "_" + conf["topology_file"] + "_"
    if conf["topology_file"] != "GEANT":
        raise ValueError("P1-A is only validated for GEANT config")
    conf.update({
        "traffic_file": "GEANT_TM1", "test_traffic_file": "GEANT_TM2",
        "train_traffic_dir": "2005_04_1st_week", "test_traffic_dir": "2005_04_2nd_week",
    })
    Path(conf["path_reward_save"]).mkdir(parents=True, exist_ok=True)
    Path(conf["path_env_save"]).mkdir(parents=True, exist_ok=True)
    return main, SimpleNamespace(**conf)


def _compat_pickle_main_classes(main_module):
    # Old samples were pickled when main.py executed under __main__. Expose
    # the same class names for compatibility, without mutating source files.
    running_main = sys.modules["__main__"]
    for name in ("InitVnetworkTrain", "InitVnetwork", "CustomMultiAgentEnv"):
        if hasattr(main_module, name) and not hasattr(running_main, name):
            setattr(running_main, name, getattr(main_module, name))


def make_mini_env_type(base_class, mini_file, mini_count):
    """Change ONLY data-file/count assignments during original __init__.

    Preserves original environment reward and transition code; avoids editing
    main.py or copying the 1.8GB full dataset for a 5-scene smoke test.
    """
    class MiniSampleEnv(base_class):
        def __init__(self, conf):
            object.__setattr__(self, "_p1a_mini_file", str(mini_file))
            object.__setattr__(self, "_p1a_mini_count", int(mini_count))
            super(MiniSampleEnv, self).__init__(conf)

        def __setattr__(self, name, value):
            if name == "sample_cnt" and hasattr(self, "_p1a_mini_count"):
                value = self._p1a_mini_count
            elif name == "sample_dict_save_file" and hasattr(self, "_p1a_mini_file"):
                value = self._p1a_mini_file
            super(MiniSampleEnv, self).__setattr__(name, value)

    return MiniSampleEnv


def create_env(args, root):
    main, config = _config(root, root / args.config)
    _compat_pickle_main_classes(main)
    if args.dataset == "mini":
        mini = (root / args.mini_file).resolve()
        mapping_file = (root / args.mini_map).resolve()
        if not mini.is_file() or not mapping_file.is_file():
            raise FileNotFoundError("Need {} and {}".format(mini, mapping_file))
        mapping_data = json.loads(mapping_file.read_text(encoding="utf-8"))
        if set(mapping_data) != set(map(str, range(len(mapping_data)))):
            raise ValueError("Mini index mapping must have consecutive keys from 0")
        original_id_by_internal = [int(mapping_data[str(i)]) for i in range(len(mapping_data))]
        if len(original_id_by_internal) != len(set(original_id_by_internal)):
            raise ValueError("Duplicate original IDs in mini mapping")
        cls = make_mini_env_type(main.CustomMultiAgentEnv, mini, len(mapping_data))
        env = cls(config)
    else:
        cls = main.CustomMultiAgentEnv
        env = cls(config)
        original_id_by_internal = list(range(len(env.sample_dict)))
    if env.sample_cnt != len(env.sample_dict):
        raise ValueError("Environment sample count mismatch")
    if len(original_id_by_internal) != env.sample_cnt:
        raise ValueError("Index mapping count mismatch")
    env.is_eval = True  # original evaluation path; no MAPPO policy used
    return env, original_id_by_internal


def select_indices(original_id_by_internal, args):
    if args.scenes.strip():
        requested = [int(s.strip()) for s in args.scenes.split(",")]
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("Scene IDs must be unique")
        lookup = {v: i for i, v in enumerate(original_id_by_internal)}
        unavailable = [s for s in requested if s not in lookup]
        if unavailable:
            raise ValueError("Missing original scene IDs: {}".format(unavailable))
        return [(lookup[s], s) for s in requested]
    if args.start < 0 or args.count < 1 or args.start + args.count > len(original_id_by_internal):
        raise ValueError("start/count exceeds dataset range")
    return [(i, original_id_by_internal[i]) for i in range(args.start, args.start + args.count)]


def load_scene_local_observations(env, internal_index):
    """Runner-only state setup; no env/global objects passed to LLM policy."""
    if internal_index not in env.sample_dict:
        raise KeyError("Missing internal sample index {}".format(internal_index))
    # Original step mutates nets/physical graph; clone avoids corrupting the
    # cached scenario when another method uses the same sample later.
    env.k1_net, env.k2_net, env.k3_net, env.g_net, env.DG = copy.deepcopy(
        env.sample_dict[internal_index]
    )
    env.sample_idx = internal_index
    for attr in ("k1_net", "k2_net", "k3_net"):
        if not hasattr(env, attr):
            raise RuntimeError("Invalid original environment")
    obs = {}
    for agent, k in zip(AGENTS[:3], (env.k1_net, env.k2_net, env.k3_net)):
        nodes, links, _ = env._get_overlap_node_link(k_net=k, g_net=env.g_net)
        obs[agent] = env._get_v_net_observations(
            k, sorted(nodes), sorted(links), True
        ).copy()
    obs["agent_g"] = env._get_g_net_observations(True).copy()
    env.observations = obs
    env.infos = {r: {} for r in AGENTS}
    # reset() only returns precomputed observations in this original environment.
    got_obs, _ = env.reset()
    if set(got_obs) != set(AGENTS):
        raise RuntimeError("Unexpected reset observation dictionary")
    return {role: np.asarray(got_obs[role], dtype=np.float32).copy() for role in AGENTS}


def _get_step_metrics(env, before_len, rewards):
    if len(env.g_done_list) != before_len + 1:
        raise RuntimeError("Exactly one g_done result must be appended per step")
    index = env.dict_eval_index
    result = {}
    for name in METRICS:
        arr = index[name]
        if len(arr) != before_len + 1:
            raise RuntimeError("Unexpected metric length for " + name)
        val = float(arr[-1])
        if not math.isfinite(val):
            raise RuntimeError("Non-finite metric: " + name)
        result[name] = val
    result["cost_sum"] = sum(result["cost_" + a] for a in ("k1", "k2", "k3"))
    result["g_done"] = bool(env.g_done_list[-1])
    result["individual_rationality"] = all(
        result["V_" + a] >= result["V_" + a + "_0"] - 1e-8
        for a in ("k1", "k2", "k3")
    ) and result["V_g"] >= -1e-8
    result["cooperative_valid"] = result["g_done"] and result["individual_rationality"]
    result["reward_mean"] = float(np.mean([float(rewards[a]) for a in AGENTS]))
    return result


def run_one_scene(env, policy, internal_index, original_index, seed):
    obs = load_scene_local_observations(env, internal_index)
    start = time.perf_counter()
    env_actions, per_role = policy.act(obs)
    policy_elapsed = time.perf_counter() - start
    # Fix the stochastic migration/forwarding costs independently of LLM latency.
    random.seed(seed + original_index)
    np.random.seed(seed + original_index)
    before = len(env.g_done_list)
    step_start = time.perf_counter()
    _, rewards, _, _, _ = env.step(env_actions)
    step_time = time.perf_counter() - step_start
    metrics = _get_step_metrics(env, before, rewards)
    return {
        "status": "ok", "original_scene_id": original_index,
        "internal_scene_index": internal_index, "metrics": metrics,
        "policy_latency_s": policy_elapsed, "env_step_s": step_time,
        "env_actions": {a: env_actions[a].tolist() for a in AGENTS},
        "per_role": per_role, "fallback_used": False,
    }


def summarize(records, policy_name):
    succeeded = [r for r in records if r["status"] == "ok"]
    errors = [r for r in records if r["status"] != "ok"]
    out = {"policy": policy_name, "n_attempted": len(records),
           "n_scored": len(succeeded), "n_failed_policy": len(errors),
           "fallback_rate": 0.0,
           "acceptance_rate_all_attempts": None,
           "cooperative_valid_rate_all_attempts": None}
    if not errors and succeeded:
        out["acceptance_rate_all_attempts"] = sum(
            r["metrics"]["g_done"] for r in succeeded) / len(records)
        out["cooperative_valid_rate_all_attempts"] = sum(
            r["metrics"]["cooperative_valid"] for r in succeeded) / len(records)
    if succeeded:
        out["acceptance_rate_scored_only"] = sum(
            r["metrics"]["g_done"] for r in succeeded) / len(succeeded)
        out["cooperative_valid_rate_scored_only"] = sum(
            r["metrics"]["cooperative_valid"] for r in succeeded) / len(succeeded)
        for key in ("V", "V_g", "V_k1", "V_k2", "V_k3", "cost_sum", "reward_mean"):
            out[key + "_scored_only"] = float(np.mean([r["metrics"][key] for r in succeeded]))
        out["mean_policy_latency_s_scored_only"] = float(np.mean(
            [r["policy_latency_s"] for r in succeeded]))
        out["mean_latency_by_role_s"] = {a: float(np.mean([
            r["per_role"][a]["model_latency_s"] for r in succeeded])) for a in AGENTS}
    return out


def run(args, *, env_override=None, mapping_override=None, client_override=None):
    root = Path.cwd()
    out = Path(args.output_dir).resolve() if args.output_dir else (
        root / "p1a_runs" / ("{}_{}_".format(args.policy, args.dataset) +
                                datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Refusing to overwrite existing run dir: {}".format(out))
    out.mkdir(parents=True, exist_ok=True)
    env, mapping = (env_override, mapping_override) if env_override is not None else create_env(args, root)
    scenes = select_indices(mapping, args)
    client = client_override or (MockClient() if args.policy == "mock" else
        ChatCompletionsClient(base_url=args.base_url, model=args.model,
                              max_tokens=args.max_tokens, temperature=args.temperature,
                              read_timeout=args.api_timeout, stream=not args.no_stream))
    policy = DecentralizedPolicy(client)
    label = "mock_not_an_llm_result" if args.policy == "mock" else "qwen_zeroshot_local"
    manifest = {
        "policy": label, "model": getattr(client, "model", args.model),
        "dataset": args.dataset, "scene_ids": [x[1] for x in scenes],
        "seed": args.seed, "config": args.config,
        "max_tokens": args.max_tokens, "temperature": args.temperature,
        "stream": not args.no_stream, "on_error": args.on_error,
        "information_boundary": "each call receives only its original per-agent observation",
        "guard_used": False, "cross_agent_communication": False,
        "task_finetuning": False, "icl_examples": 0, "fallback_used": False,
        "main_py_sha256": _hash_file(root / "main.py"),
        "notes": "P1-A development run, not final independent held-out evaluation",
    }
    _json_write(out / "manifest.json", manifest)
    records = []
    try:
        with (out / "records.jsonl").open("w", encoding="utf-8") as f:
            for internal, original in scenes:
                try:
                    record = run_one_scene(env, policy, internal, original, args.seed)
                except PolicyFailure as exc:
                    record = {"status": "policy_error", "original_scene_id": original,
                              "internal_scene_index": internal, "failed_role": exc.role,
                              "error": str(exc), "partial_decisions": exc.partial,
                              "fallback_used": False}
                records.append(record)
                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                f.flush()
                if record["status"] == "ok":
                    m = record["metrics"]
                    print("scene={} admission={} IR+admit={} V={:.3f} policy_time={:.2f}s".format(
                        original, int(m["g_done"]), int(m["cooperative_valid"]),
                        m["V"], record["policy_latency_s"]), flush=True)
                else:
                    print("scene={} FAILED role={} (no env.step)".format(
                        original, record["failed_role"]), flush=True)
                    if args.on_error == "abort":
                        raise RuntimeError(record["error"])
    finally:
        summary = summarize(records, label)
        _json_write(out / "summary.json", summary)
        with (out / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["original_scene_id", "status", "g_done",
                "cooperative_valid", "individual_rationality", "V", "V_g", "V_k1",
                "V_k2", "V_k3", "cost_sum", "reward_mean", "policy_latency_s",
                "env_step_s", "failed_role"])
            writer.writeheader()
            for rec in records:
                row = {"original_scene_id": rec["original_scene_id"], "status": rec["status"],
                       "failed_role": rec.get("failed_role", "")}
                if rec["status"] == "ok":
                    row.update({key: rec["metrics"].get(key) for key in
                        ("g_done", "cooperative_valid", "individual_rationality", "V",
                         "V_g", "V_k1", "V_k2", "V_k3", "cost_sum", "reward_mean")})
                    row["policy_latency_s"] = rec["policy_latency_s"]
                    row["env_step_s"] = rec["env_step_s"]
                writer.writerow(row)
        print("Output directory: {}".format(out), flush=True)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    run(get_args())
