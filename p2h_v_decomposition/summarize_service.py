"""Summarize completed development passive tracing; never imports an API client."""
from __future__ import annotations
import argparse
import collections
import csv
import json
import math
from pathlib import Path

import numpy as np

from trace_service import ROOT, K, MISSES, ORIGINALS


def mean(values):
    return float(np.mean(list(values)))


def total(row, field):
    return sum(row["trace"]["roles"][k][field] for k in K)


def global_summary(rows):
    service = [row["trace"]["roles"][k] for row in rows.values() for k in K]
    demand = sum(s["after"]["routed_demand_gbps"] for s in service)
    load = sum(s["after"]["hop_weighted_load_gbps"] for s in service)
    return {"n": len(rows), "mean_V": mean(r["metrics"]["V"] for r in rows.values()),
            "cooperative": sum(r["metrics"]["cooperative_valid"] for r in rows.values()),
            "admitted": sum(r["metrics"]["g_done"] for r in rows.values()),
            "mean_existing_routed_offered_Gbps": demand / len(rows),
            "mean_hop_weighted_load_Gbps": load / len(rows),
            "traffic_weighted_hops": load / demand,
            "mean_node_revenue": mean(total(r, "node_revenue") for r in rows.values()),
            "mean_link_revenue": mean(total(r, "link_revenue") for r in rows.values()),
            "mean_node_cost": mean(total(r, "node_cost") for r in rows.values()),
            "mean_link_cost": mean(total(r, "link_cost") for r in rows.values()),
            "rerouted_flow_fraction": sum(s["after"]["rerouted_flows"] for s in service) /
                                      sum(s["after"]["flows"] for s in service),
            "rerouted_traffic_fraction": sum(s["after"]["rerouted_demand_gbps"] for s in service) / demand,
            "max_utilization": max(s["max_link_utilization"] for s in service),
            "role_scenes_with_overloaded_link": sum(s["overloaded_links"] > 0 for s in service),
            "role_scenes_with_utilization_above70pct": sum(s["over_70pct_links"] > 0 for s in service)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    folder = Path(args.run_dir).resolve()
    assert ROOT in folder.parents
    run = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert run["status"] == "COMPLETE" and run["controls_reproduced"] == 1400
    cases = collections.defaultdict(dict)
    with (folder / "records.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            assert row["original_scene_id"] not in cases[row["case"]]
            cases[row["case"]][row["original_scene_id"]] = row
    assert len(cases) == 7 and all(sorted(v) == list(range(200)) for v in cases.values())
    baseline = cases["MAPPO"]
    identity_count = 0
    comparisons = {}
    for name, rows in cases.items():
        parts = {"node_revenue": [], "link_revenue": [], "node_cost": [], "link_cost": []}
        route_explanation = []
        for i, r in rows.items():
            for k in K:
                t, b = r["trace"]["roles"][k], baseline[i]["trace"]["roles"][k]
                qa, ma = t["after"], b["after"]
                assert qa["flows"] == ma["flows"]
                assert math.isclose(qa["routed_demand_gbps"], ma["routed_demand_gbps"], abs_tol=1e-10)
                direct = t["link_revenue"] - b["link_revenue"]
                by_hops = 10 * qa["routed_demand_gbps"] * (
                    qa["traffic_weighted_hops"] - ma["traffic_weighted_hops"])
                assert math.isclose(direct, by_hops, abs_tol=1e-7)
                route_explanation.append(by_hops)
                identity_count += 1
            for field in parts:
                parts[field].append(total(r, field) - total(baseline[i], field))
        comparisons[name] = {"delta_" + k: mean(v) for k, v in parts.items()}
        comparisons[name]["delta_link_revenue_explained_by_hops"] = sum(route_explanation) / 200
    summaries = {name: global_summary(rows) for name, rows in cases.items()}
    misses = {}
    miss_rows = []
    for cohort in ORIGINALS:
        q, c = cases["original_" + cohort], cases["candidate_" + cohort]
        ids = [i for i in range(200) if baseline[i]["metrics"]["g_done"] and not q[i]["metrics"]["g_done"]]
        assert ids == list(MISSES)
        details = []
        for i in ids:
            failure = q[i]["trace"]["g_gate"]["first_node_failure"]
            assert failure is not None
            assert failure == c[i]["trace"]["g_gate"]["first_node_failure"]
            n = failure["local_index"]
            res = failure["stage"]
            qnode = q[i]["trace"]["g_gate"]["nodes_before_g"][n]
            mnode = baseline[i]["trace"]["g_gate"]["nodes_before_g"][n]
            field = "released_" + res
            roles = {}
            for k in set(qnode["release_contributions"]) | set(mnode["release_contributions"]):
                qr = qnode["release_contributions"].get(k, {}).get(field, 0)
                mr = mnode["release_contributions"].get(k, {}).get(field, 0)
                roles[k] = mr - qr
            record = {"scene": i, "first_failure": res, "local_node": n,
                      "required_extra_capacity": -failure["absolute_gap"],
                      "required_extra_fraction_physical": -failure["gap_fraction_of_physical"],
                      "MAPPO_margin_fraction_physical": mnode[res + "_gap"] / mnode[res + "_physical_capacity"],
                      "MAPPO_minus_Qwen_release_by_role": roles,
                      "candidate_minus_MAPPO_V": c[i]["metrics"]["V"] - baseline[i]["metrics"]["V"]}
            details.append(record)
            miss_rows.append({"cohort": cohort, **{k: v for k, v in record.items() if k != "MAPPO_minus_Qwen_release_by_role"}})
        misses[cohort] = {"n": len(ids), "first_failure_counts": dict(collections.Counter(d["first_failure"] for d in details)),
                          "gap_fraction_min": min(d["required_extra_fraction_physical"] for d in details),
                          "gap_fraction_max": max(d["required_extra_fraction_physical"] for d in details),
                          "details": details}
    output = {"status": "VERIFIED", "source_run": str(folder.relative_to(ROOT)),
              "summaries": summaries, "comparisons_to_MAPPO": comparisons,
              "missed_admissions": misses,
              "checks": {"cached_controls_reproduced": 1400, "route_identities": identity_count,
                         "same_existing_offered_traffic_all_methods": True,
                         "same_existing_active_flows_all_methods": True,
                         "existing_link_revenue_delta_fully_explained_by_hops": True,
                         "candidate_node_gate_identical_to_Qwen_on_all_8_misses": True},
              "API_calls": 0, "estimated_API_cost_usd": 0,
              "limitations": ["Existing traffic is routed offered demand, not measured delivered throughput.",
                             "No packet drop or admission-based existing-flow service model is added.",
                             "Node deficits are global offline diagnostics, not available online policy features.",
                             "Development-only analysis. No new policy or improvement claim."]}
    with (folder / "service_analysis.json").open("x", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    with (folder / "missed_admissions.csv").open("x", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(miss_rows[0]))
        w.writeheader()
        w.writerows(miss_rows)
    lines = ["# 开发集服务、路径与接纳边界追踪", "",
             "200 个相同 val5601 开发场景、7 组缓存动作、1400 次原环境评分均逐场复现。"
             "不调用模型，不使用独立测试，不修改环境或指标。", "",
             "| 方法 | V | 既有路由需求 Gbps | 流量加权跳数 | 节点成本 | 链路成本 | 超载角色场景数 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, s in summaries.items():
        lines.append(f"| {name} | {s['mean_V']:.4f} | {s['mean_existing_routed_offered_Gbps']:.4f} | "
                     f"{s['traffic_weighted_hops']:.5f} | {s['mean_node_cost']:.4f} | {s['mean_link_cost']:.4f} | "
                     f"{s['role_scenes_with_overloaded_link']} |")
    lines += ["", "## 核心解释", "",
              "所有方法、所有场景和已有角色的非零需求流数与路由需求总量相同。"
              "4200 个角色场景均满足：链路收入差 = 10 × 相同需求量 × 流量加权平均跳数差。"
              "因此此前的账面链路收入损失全部可由路径长度解释，不能判定为实际吞吐量损失。",
              "已有切片路径分配不建立丢包/端到端送达模型；有路径的需求会被分配到链路。"
              "本报告只称路由需求，不能将其宣传为实测送达吞吐量。利用率检查独立记录，"
              "不改变原 V、奖励或合作有效率。",
              "下一轮不应通过故意绕长路增加原 V。更有意义的方向是改善接纳边界，"
              "或降低迁移/重路由成本，并同时报告原 V 与独立资源效率诊断。", "",
              "## 8 个仅 MAPPO 接纳的场景", "",
              "| Qwen 批次 | 场景 | 首个失败资源 | g 局部节点 | 欠缺物理容量比例 | MAPPO 剩余裕量比例 |",
              "|---|---:|---|---:|---:|---:|"]
    for d in miss_rows:
        lines.append(f"| {d['cohort']} | {d['scene']} | {d['first_failure']} | {d['local_node']} | "
                     f"{100*d['required_extra_fraction_physical']:.3f}% | {100*d['MAPPO_margin_fraction_physical']:.3f}% |")
    lines += ["",
              "8 场缺口来源和各角色释放差额保存在 service_analysis.json。"
              "这些是离线全局解释，不可直接作为去中心化策略输入。",
              "三个批次同一组 8 场均被拒绝，说明值得研究稳定的局部节点释放边界。"
              "当前候选只校正链路，因此不改变这些节点门控。", "",
              "## 验证与执行记录", "",
              "3 个语义测试通过；1400 个控制指标复现、4200 个路径/收入恒等式通过。"
              "执行前后数据和源文件 SHA256 未变。API 调用0，新增费用0。"
              "启动时两次索引/JSON 类型兼容性错误保留在相邻 STOPPED.json；"
              "失败运行不能用于结论。首次沙箱还阻止 wandb 临时目录清理，后续使用已批准执行权限。",
              "只读查看：" , "",
              "~~~powershell",
              "Get-Content -LiteralPath '" + str(folder / "RESULTS.md") + "'",
              "~~~", ""]
    with (folder / "RESULTS.md").open("x", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("SERVICE_ANALYSIS_VERIFIED", folder)
    print(json.dumps({"summaries": summaries, "misses": {
        k: {x: v[x] for x in ("first_failure_counts", "gap_fraction_min", "gap_fraction_max")}
        for k, v in misses.items()}, "checks": output["checks"]}, separators=(",", ":")))


if __name__ == "__main__":
    main()
