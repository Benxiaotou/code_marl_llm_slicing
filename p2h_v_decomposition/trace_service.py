"""Development-only passive route/service/admission tracing of cached actions."""
from __future__ import annotations
import contextlib
import ctypes
import datetime as dt
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "p2f_analysis"))
import val5601_counterfactual_replay as replay

MULTI = ROOT / "p2g_autonomous/results/teacher_multicohort_20261104T164508_605375Z/records.jsonl"
ORIGINALS = {
    "historical": (ROOT / "p2b_runs/val5601_standard1000_p2a200/records.jsonl", None),
    "fresh_linkrun": (ROOT / "p2f_analysis/qwen_full200/run_20261104T141915_396460Z/full_original_records.jsonl", None),
    "fresh_noderun": (ROOT / "p2g_autonomous/results/node_full_20261104T161150_600593Z/records.jsonl", "original"),
}
MISSES = (32, 45, 110, 126, 127, 133, 187, 196)
K = ("k1", "k2", "k3")


def path_load(demand, paths):
    """Traffic counted per traversed link; not a throughput/drop estimator."""
    return float(demand) * sum(len(p) - 1 for p in paths) / len(paths)


def first_node_failure(rows):
    for row in rows:
        for resource in ("compute", "cache"):
            gap = row[resource + "_gap"]
            if gap < 0:
                return {"stage": resource, "local_index": row["local_index"],
                        "absolute_gap": gap, "gap_fraction_of_physical":
                        gap / row[resource + "_physical_capacity"]}
    return None


def windows_ram_gib():
    class Memory(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
            (k, ctypes.c_ulonglong) for k in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile",
                                             "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual",
                                             "ullAvailExtendedVirtual")]
    stat = Memory()
    stat.dwLength = ctypes.sizeof(stat)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        raise OSError("GlobalMemoryStatusEx failed")
    return stat.ullAvailPhys / 1024 ** 3


def selected_row(row):
    return {"metrics": row["metrics"], "env_actions": row["env_actions"]}


def load_rows(path, arm=None):
    data = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if arm and row["arm"] != arm:
                continue
            i = row.get("original_scene_id", row.get("scene"))
            assert i not in data
            data[i] = selected_row(row)
    assert sorted(data) == list(range(200))
    return data


class ServiceTrace:
    """Wrappers inspect inputs/returns and forward all original results unchanged."""
    def __init__(self, env):
        self.env = env
        self.reset()
        node_original = env._get_k_node_cost
        link_original = env._get_k_link_cost
        route_original = env._ksp_traffic_distribution
        g_original = env._get_g_adjustment

        def node(k_net, idx_overlap_nodes, action_k_node):
            result = node_original(k_net, idx_overlap_nodes, action_k_node)
            self.node_costs.append(float(result[0]) * .005)
            return result

        def route(v_net, link_loads, demand, src, dst, paths):
            if self.active is not None:
                c = self.active
                phase = "before" if c["calls"] < c["n_active"] else "after"
                stats = c[phase]
                stats["flows"] += 1
                stats["routed_demand_gbps"] += float(demand)
                stats["hop_weighted_load_gbps"] += path_load(demand, paths)
                if phase == "after":
                    old = v_net.shortest_paths_node[v_net.pair_sd_to_idx[(src, dst)]]
                    changed = tuple(map(tuple, paths)) != tuple(map(tuple, old))
                    stats["rerouted_flows"] += int(changed)
                    stats["rerouted_demand_gbps"] += float(demand) * int(changed)
                c["calls"] += 1
            return route_original(v_net, link_loads, demand, src, dst, paths)

        def link(k_net, idx_overlap_links, action_k_link, tm_idx):
            tm = k_net.traffic_matrices[tm_idx] / 1e6
            n = sum(tm[k_net.pair_idx_to_sd[i][0]][k_net.pair_idx_to_sd[i][1]] != 0
                    for i in range(k_net.num_pairs))
            def empty():
                return {"flows": 0, "routed_demand_gbps": 0., "hop_weighted_load_gbps": 0.,
                        "rerouted_flows": 0, "rerouted_demand_gbps": 0.}
            self.active = {"n_active": int(n), "calls": 0, "before": empty(), "after": empty()}
            try:
                result = link_original(k_net, idx_overlap_links, action_k_link, tm_idx)
                trace = self.active
                assert trace["calls"] == 2 * n
                for phase in ("before", "after"):
                    s = trace[phase]
                    s["traffic_weighted_hops"] = s["hop_weighted_load_gbps"] / s["routed_demand_gbps"]
                assert math.isclose(trace["before"]["hop_weighted_load_gbps"], result[4], abs_tol=1e-8)
                assert math.isclose(trace["after"]["hop_weighted_load_gbps"], result[3], abs_tol=1e-8)
                assert math.isclose(trace["before"]["routed_demand_gbps"],
                                    trace["after"]["routed_demand_gbps"], abs_tol=1e-10)
                links = [k_net.DG[k_net.link_idx_to_sd[i][0]][k_net.link_idx_to_sd[i][1]]
                         for i in range(k_net.num_links)]
                util = [e["used_bandw"] / e["bandwidth"] for e in links]
                trace["max_link_utilization"] = float(max(util))
                trace["overloaded_links"] = int(sum(u > 1. for u in util))
                trace["over_70pct_links"] = int(sum(u > .7 for u in util))
                trace["link_cost"] = .2 * float(result[0])
                trace["link_revenue"] = 10 * float(result[3])
                trace.pop("calls")
                self.links.append(trace)
            finally:
                self.active = None
            return result

        def g(g_net, action_g, assigned_resoure, tm_idx=0):
            rows = []
            for i in range(g_net.num_nodes):
                pi = env.node_id_to_idx[g_net.node_id_list[i]]
                gn, pn = g_net.DG.nodes[i], env.DG.nodes[pi]
                rec = {"local_index": i, "release_contributions": {}}
                for name, cap, alloc in (("compute", "comp_capacity", "allocated_comp"),
                                          ("cache", "cach_capacity", "allocated_cach")):
                    rec[name + "_physical_capacity"] = float(pn[cap])
                    rec[name + "_free"] = float(pn[cap] - pn[alloc])
                    rec[name + "_original_g_request"] = float(gn[cap])
                    rec[name + "_gap"] = float(pn[cap] - pn[alloc] - gn[cap])
                for k in K:
                    kn = getattr(env, k + "_net")
                    if g_net.node_id_list[i] in kn.node_id_list:
                        local = list(kn.node_id_list).index(g_net.node_id_list[i])
                        rc, rh, _ = assigned_resoure[k + "_net"]
                        rec["release_contributions"][k] = {
                            "local_node": local, "released_compute": float(rc.get(local, 0)),
                            "released_cache": float(rh.get(local, 0))}
                rows.append(rec)
            first = first_node_failure(rows)
            result = g_original(g_net, action_g, assigned_resoure, tm_idx)
            if first:
                assert not result[3]
            self.g = {"nodes_before_g": rows, "first_node_failure": first,
                      "admitted": bool(result[3]), "demand_n_counted": float(result[1]),
                      "demand_l_counted": float(result[2])}
            return result

        env._get_k_node_cost = node
        env._get_k_link_cost = link
        env._ksp_traffic_distribution = route
        env._get_g_adjustment = g

    def reset(self):
        self.links = []
        self.node_costs = []
        self.g = None
        self.active = None

    def finish(self, metrics):
        assert len(self.links) == len(self.node_costs) == 3
        out = {"roles": {}, "g_gate": self.g}
        for i, k in enumerate(K):
            trace = self.links[i]
            trace["node_cost"] = self.node_costs[i]
            trace["node_revenue"] = metrics["revenue_" + k] - trace["link_revenue"]
            assert math.isclose(trace["node_cost"] + trace["link_cost"], metrics["cost_" + k], abs_tol=1e-8)
            out["roles"][k] = trace
        return out


def main():
    import p1a_runner as base
    import p1c_runner
    from p2e_prompt.val5601_eval import verify_environment
    inputs = [ROOT / "p1b_data/p2b_val_seed5601.json", ROOT / "p2b_runs/val5601_mappo200/records.jsonl",
              MULTI, *(p for p, _ in ORIGINALS.values())]
    for p in inputs:
        assert p.is_file()
    manifest = json.loads(inputs[0].read_text(encoding="utf-8"))
    assert manifest["split"] == "validation"
    sample = Path(manifest["sample_file"])
    assert sample.is_file() and sample.stat().st_size == 644537810
    ram = windows_ram_gib()
    if ram < 6:
        raise RuntimeError("Insufficient RAM; no replay started")
    assert shutil.disk_usage(ROOT).free > 1024 ** 3
    cases = {"MAPPO": load_rows(inputs[1])}
    for label, (path, arm) in ORIGINALS.items():
        cases["original_" + label] = load_rows(path, arm)
    for label in ORIGINALS:
        cases["candidate_" + label] = {}
    with MULTI.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["method"] == "link_mean3_a0.5":
                cases["candidate_" + r["cohort"]][r["original_scene_id"]] = selected_row(r)
    assert all(sorted(x) == list(range(200)) for x in cases.values())
    out = ROOT / "p2h_v_decomposition" / ("service_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    out.mkdir(exist_ok=False)
    runtime = out / "env_runtime"
    runtime.mkdir()
    hashes = {str(p.relative_to(ROOT)): replay.sha256(p) for p in
              (*inputs, ROOT / "main.py", ROOT / "configs_main.yaml", Path(__file__).resolve())}
    environment = verify_environment(ROOT, manifest)
    protocol = {"schema": "val5601_passive_route_gate_v1", "dataset": manifest["dataset_id"],
                "split": "validation", "scene_ids": list(range(200)), "seed": 1,
                "cases": list(cases), "expected_scores": 1400, "API_calls": 0,
                "available_RAM_GiB": ram, "sample_bytes": sample.stat().st_size,
                "sources_sha256": hashes, "environment": environment,
                "instrumentation": "Return original outputs unchanged, no added RNG calls, reload before each step.",
                "limitations": "Routed offered traffic is not measured delivered throughput. Original existing-slice routing does not model packet drops."}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    count = 0
    with (out / "env_stdout.log").open("x", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log):
            mm, _ = base._config(ROOT, ROOT / "configs_main.yaml")
            base._compat_pickle_main_classes(mm)
            env = p1c_runner.create_eval_env(ROOT, manifest, "configs_main.yaml")
        env.path_reward_save = str(runtime)
        env.path_env_save = str(runtime)
        trace = ServiceTrace(env)
        try:
            with (out / "records.jsonl").open("x", encoding="utf-8") as sink:
                for case, source in cases.items():
                    for scene in range(200):
                        trace.reset()
                        with contextlib.redirect_stdout(log):
                            base.load_scene_local_observations(env, scene)
                            random.seed(1 + scene)
                            np.random.seed(1 + scene)
                            before = len(env.g_done_list)
                            _, rewards, _, _, _ = env.step({r: np.asarray(v, dtype=np.float32)
                                                          for r, v in source[scene]["env_actions"].items()})
                            metrics = base._get_step_metrics(env, before, rewards)
                        diff = replay.control_diff(source[scene]["metrics"], metrics)
                        if diff:
                            raise RuntimeError("CONTROL_MISMATCH " + case + "/" + str(scene) + " " + repr(diff))
                        row = {"case": case, "original_scene_id": scene, "metrics": metrics,
                               "trace": trace.finish(metrics)}
                        sink.write(json.dumps(row, allow_nan=False) + "\n")
                        count += 1
                    sink.flush()
                    os.fsync(sink.fileno())
                    print("CASE_VALIDATED", case, flush=True)
            for p, h in hashes.items():
                assert replay.sha256(ROOT / p) == h
            summary = {"status": "COMPLETE", "controls_reproduced": count,
                       "source_hashes_unchanged": True, "API_calls": 0, "estimated_API_cost_usd": 0}
            (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print("SERVICE_OUTPUT", out, flush=True)
        except Exception as exc:
            (out / "STOPPED.json").write_text(json.dumps({"case": case, "scene": scene,
                 "completed": count, "error": str(exc)}, indent=2), encoding="utf-8")
            raise


if __name__ == "__main__":
    main()
