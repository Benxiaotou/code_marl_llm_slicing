"""Read-only accounting decomposition of saved val5601 development metrics.

No environment construction, policy selection, model client, or independent
test inputs. Outputs are new directories, and every accounting identity is
checked per scene before groups are interpreted.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MULTI = ROOT / "p2g_autonomous/results/teacher_multicohort_20261104T164508_605375Z/records.jsonl"
TRACE = ROOT / "p2f_analysis/val5601_trace/run_20261104T131408_665359Z"
COHORTS = {
    "historical": ("p2b_runs/val5601_standard1000_p2a200/records.jsonl", None),
    "fresh_linkrun": ("p2f_analysis/qwen_full200/run_20261104T141915_396460Z/full_original_records.jsonl", None),
    "fresh_noderun": ("p2g_autonomous/results/node_full_20261104T161150_600593Z/records.jsonl", "original"),
}
GROUPS = ("both_accepted", "MAPPO_only_accepted", "Qwen_only_accepted", "both_rejected")
K = ("k1", "k2", "k3")
FIELDS = ("V", "V_g", "g_done", "cooperative_valid", "individual_rationality", "cost_sum",
          "revenue_g") + tuple(f"{x}_{k}" for k in K for x in
                              ("revenue", "cost", "V", "X")) + tuple(f"V_{k}_0" for k in K)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def close(a, b):
    assert math.isclose(float(a), float(b), abs_tol=1e-7, rel_tol=1e-9), (a, b)


def validate(m):
    assert all(math.isfinite(float(m[k])) for k in FIELDS)
    close(m["cost_sum"], sum(m["cost_" + k] for k in K))
    close(m["V"], sum(m["revenue_" + k] for k in K) + m["revenue_g"] - m["cost_sum"])
    close(m["V"], sum(m["V_" + k] for k in K) + m["V_g"])
    for k in K:
        close(m["V_" + k], m["revenue_" + k] - m["cost_" + k] + m["X_" + k])
    close(m["V_g"], m["revenue_g"] - sum(m["X_" + k] for k in K))
    ir = m["V_g"] >= -1e-8 and all(m["V_" + k] >= m["V_" + k + "_0"] - 1e-8 for k in K)
    assert bool(m["individual_rationality"]) == ir
    assert bool(m["cooperative_valid"]) == (bool(m["g_done"]) and ir)


def read_metrics(path, arm=None):
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if arm and r["arm"] != arm:
                continue
            i = r.get("original_scene_id", r.get("scene"))
            assert i not in out
            out[i] = {k: r["metrics"][k] for k in FIELDS}
            validate(out[i])
    assert sorted(out) == list(range(200))
    return out


def group(m, q):
    return { (True, True): GROUPS[0], (True, False): GROUPS[1],
             (False, True): GROUPS[2], (False, False): GROUPS[3] }[
                 bool(m["g_done"]), bool(q["g_done"])]


def components(m):
    return {"V": m["V"], "existing_revenue": sum(m["revenue_" + k] for k in K),
            "new_slice_revenue": m["revenue_g"], "minus_cost": -m["cost_sum"],
            **{"minus_cost_" + k: -m["cost_" + k] for k in K},
            **{"revenue_" + k: m["revenue_" + k] for k in K},
            "V_g": m["V_g"]}


def delta(a, b):
    ac, bc = components(a), components(b)
    out = {k: ac[k] - bc[k] for k in ac}
    close(out["V"], out["existing_revenue"] + out["new_slice_revenue"] + out["minus_cost"])
    return out


def describe(ids, q, m, denominator=200):
    diffs = [delta(q[i], m[i]) for i in ids]
    sums = {k: sum(d[k] for d in diffs) for k in components(m[0])}
    contribution = {k: v / denominator for k, v in sums.items()}
    return {"n": len(ids), "scene_ids": ids,
            "delta_within_group_mean": {k: v / len(ids) if ids else None for k, v in sums.items()},
            "contribution_to_all_scene_mean": contribution,
            "V_wins": sum(d["V"] > 1e-7 for d in diffs),
            "V_losses": sum(d["V"] < -1e-7 for d in diffs),
            "qwen_cooperative": sum(bool(q[i]["cooperative_valid"]) for i in ids),
            "MAPPO_cooperative": sum(bool(m[i]["cooperative_valid"]) for i in ids),
            "qwen_admitted_ir_invalid": sum(q[i]["g_done"] and not q[i]["individual_rationality"] for i in ids),
            "MAPPO_admitted_ir_invalid": sum(m[i]["g_done"] and not m[i]["individual_rationality"] for i in ids)}


def decomposition(q, m):
    groups = {name: describe([i for i in range(200) if group(m[i], q[i]) == name], q, m)
              for name in GROUPS}
    overall = describe(list(range(200)), q, m)
    for key in overall["contribution_to_all_scene_mean"]:
        close(sum(g["contribution_to_all_scene_mean"][key] for g in groups.values()),
              overall["contribution_to_all_scene_mean"][key])
    ids = [i for i in range(200) if m[i]["g_done"] and q[i]["g_done"]]
    ir_strata = {
        name: describe([i for i in ids if bool(q[i]["cooperative_valid"]) == flag], q, m)
        for name, flag in (("both_admitted_Qwen_IR_valid", True), ("both_admitted_Qwen_IR_invalid", False))
    }
    rng = np.random.default_rng(20261105)
    ds = np.array([q[i]["V"] - m[i]["V"] for i in range(200)])
    ci = np.quantile(ds[rng.integers(0, 200, (10000, 200))].mean(axis=1), [.025, .975])
    return {"groups": groups, "overall": overall, "both_admitted_IR_strata": ir_strata,
            "V_paired_bootstrap_CI95_exploratory": ci.tolist()}


def passive_historical(m, h):
    trace = {}
    for label, tag, source in (("MAPPO", "MM", m), ("historical_Qwen", "QQ", h)):
        rows = {}
        with (TRACE / (tag + "_records.jsonl")).open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                i = r["original_scene_id"]
                for k in FIELDS:
                    close(r["metrics"][k], source[i][k])
                t = r["passive_trace"]
                c = {key: sum(t["roles"][role][key] for role in K)
                     for key in ("node_cost", "link_cost", "node_revenue", "link_revenue")}
                close(c["node_cost"] + c["link_cost"], source[i]["cost_sum"])
                close(c["node_revenue"] + c["link_revenue"], sum(source[i]["revenue_" + k] for k in K))
                rows[i] = c
        assert sorted(rows) == list(range(200))
        trace[label] = rows
    return {name: {
        "n": sum(group(m[i], h[i]) == name for i in range(200)),
        "contribution_to_all_scene_mean": {
            key: sum(trace["historical_Qwen"][i][key] - trace["MAPPO"][i][key]
                     for i in range(200) if group(m[i], h[i]) == name) / 200
            for key in ("node_cost", "link_cost", "node_revenue", "link_revenue")}}
        for name in GROUPS}


def main():
    manifest_path = ROOT / "p1b_data/p2b_val_seed5601.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset_id"] == "p2b_val_seed5601" and manifest["split"] == "validation"
    mpath = ROOT / "p2b_runs/val5601_mappo200/records.jsonl"
    paths = [manifest_path, mpath, MULTI, *(ROOT / v[0] for v in COHORTS.values()),
             TRACE / "MM_records.jsonl", TRACE / "QQ_records.jsonl", Path(__file__).resolve()]
    hashes = {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": digest(p)}
              for p in paths}
    m = read_metrics(mpath)
    original = {k: read_metrics(ROOT / p, arm) for k, (p, arm) in COHORTS.items()}
    candidate = {k: {} for k in COHORTS}
    with MULTI.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["method"] != "link_mean3_a0.5":
                continue
            c, i = r["cohort"], r["original_scene_id"]
            assert i not in candidate[c]
            candidate[c][i] = {k: r["metrics"][k] for k in FIELDS}
            validate(candidate[c][i])
    analyses = {}
    for c in COHORTS:
        assert sorted(candidate[c]) == list(range(200))
        for i in range(200):
            for key in ("V_k1_0", "V_k2_0", "V_k3_0"):
                close(m[i][key], original[c][i][key]); close(m[i][key], candidate[c][i][key])
            assert bool(original[c][i]["g_done"]) == bool(candidate[c][i]["g_done"])
        analyses[c] = {"original_Qwen_vs_MAPPO": decomposition(original[c], m),
                       "frozen_candidate_vs_MAPPO": decomposition(candidate[c], m),
                       "candidate_minus_original": describe(list(range(200)), candidate[c], original[c])}
    trace = passive_historical(m, original["historical"])
    report = {"status": "COMPLETE", "dataset_id": manifest["dataset_id"], "split": "validation",
              "n_unique_scenes": 200, "n_Qwen_draws": 3, "three_draws_not_600_independent_scenes": True,
              "difference_sign": "Qwen/candidate minus MAPPO; minus_cost positive means savings",
              "analyses": analyses, "historical_passive_trace": trace,
              "verification": {"metrics_rows_checked": 1400, "trace_rows_checked": 400,
                               "accounting_and_IR_identities": True, "groups_additive": True,
                               "paired_scene_baselines": True, "candidate_admission_unchanged": True,
                               "no_environment_steps": True, "no_API_calls": True},
              "inputs": hashes, "estimated_API_cost_usd": 0,
              "limitations": ["Repeatedly used validation scenes; exploratory subgroup accounting.",
                             "Passive node/link trace exists only for historical Qwen, not candidate/fresh draws.",
                             "Existing link revenue is route-length-weighted, not end-to-end throughput.",
                             "Admission-group decomposition is descriptive and not a causal policy ablation."]}
    out = ROOT / "p2h_v_decomposition" / ("report_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    out.mkdir(parents=True, exist_ok=False)
    with (out / "analysis.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n")
    lines = ["# val5601 开发集 V 收益与损失分解", "",
             "200 个相同开发场景，3 次已保存 Qwen 采样；无新推理、无环境评分、无测试集读取。",
             "下表均为对全部 200 场景平均 V 差异的贡献，各行可直接相加。正数表示优于 MAPPO。", "",
             "| Qwen 批次 / 方法 | 分组 | 场景数 | V 贡献 | 已有收入贡献 | 新切片收入贡献 | 成本节省贡献 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    names = {"both_accepted": "双方接纳", "MAPPO_only_accepted": "仅 MAPPO 接纳",
             "Qwen_only_accepted": "仅 Qwen 接纳", "both_rejected": "双方拒绝"}
    for cohort, analysis in analyses.items():
        for arm in ("original_Qwen_vs_MAPPO", "frozen_candidate_vs_MAPPO"):
            for name, g in analysis[arm]["groups"].items():
                x = g["contribution_to_all_scene_mean"]
                lines.append(f"| {cohort}/{arm} | {names[name]} | {g['n']} | {x['V']:.4f} | "
                             f"{x['existing_revenue']:.4f} | {x['new_slice_revenue']:.4f} | {x['minus_cost']:.4f} |")
            x = analysis[arm]["overall"]["contribution_to_all_scene_mean"]
            lines.append(f"| {cohort}/{arm} | 全部 | 200 | {x['V']:.4f} | {x['existing_revenue']:.4f} | "
                         f"{x['new_slice_revenue']:.4f} | {x['minus_cost']:.4f} |")
    lines += ["", "## 已有被动追踪：历史 Qwen 对 MAPPO", "",
              "仅适用于历史动作批次，不将其当作新鲜批次或候选的逐项追踪。正成本表示 Qwen 支出更多。", "",
              "| 分组 | 数量 | 节点收入差贡献 | 链路收入差贡献 | 节点成本差贡献 | 链路成本差贡献 |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, g in trace.items():
        x = g["contribution_to_all_scene_mean"]
        lines.append(f"| {names[name]} | {g['n']} | {x['node_revenue']:.4f} | {x['link_revenue']:.4f} | "
                     f"{x['node_cost']:.4f} | {x['link_cost']:.4f} |")
    lines += ["", "## 结论与下一阶段建议", "",
              "1. 三批 Qwen 动作只对应 200 个相同开发场景，不能当作 600 个独立样本。"
              "候选 V 对 MAPPO 平均差分别 +3.8515、+4.1981、+3.7917，"
              "配对 Bootstrap 区间均跨 0，因此不构成显著胜出的证明。",
              "2. 双方接纳的 152 场仍存在稳定账面损失，每场平均 V 低约 2.43～2.77。"
              "原有收入下降比成本节省更大，并非总成本偏高造成这组净损失。",
              "3. 每批仅 MAPPO 接纳的同 8 场，每场损失约 80.8～84.0；"
              "仅 Qwen 接纳的 26～27 场每场增加约 72.5～75.0。"
              "总体优势依赖额外接纳收益抵消双方接纳场景中的损失，提示存在泛化敏感性，"
              "但本分解不能证明独立测试落后的具体原因。",
              "4. 历史被动追踪中，已有节点收入差全体平均 -0.2150，"
              "链路收入差 -7.1024；节点成本差 +0.4949、链路成本差 -1.0483。"
              "此细分严格限于历史 Qwen，不能移用于候选或两个新鲜采样。",
              "5. 下一步先进行只读服务/路由追踪：对照实际服务流量、流是否成功送达、"
              "路径跳数与重路由成本，判断账面链路收入差的来源。"
              "原指标中链路收入按路径累计，收入变低不能直接解释为吞吐量降低；"
              "不得通过故意绕路制造 V 改善。",
              "6. 随后只针对开发集，诊断 8 场 MAPPO 接纳而 Qwen 拒绝的局部节点容量边界；"
              "任何修正必须同时检查原有 Qwen 额外接纳场景，避免把新增收益丢掉。"
              "当前不改变提示词、动作策略或环境数学定义。",
              "", "## 本次执行与复现", "",
              "仅新增本目录中的分析脚本和新报告文件。没有 API、环境评分、训练、"
              "原始数据修改或独立测试读取。1400 条指标及 400 条历史追踪通过会计恒等式、"
              "IR、成对基线与分组加总核验，所有输入文件执行前后 SHA256 一致。",
              "第一次启动仅遇到脚本自身相对路径处理错误，在读取指标前退出；"
              "修正为绝对路径后完整通过。",
              "下一步只读查看命令（不启动实验）：", "", "~~~powershell",
              "Get-Content -LiteralPath '" + str(out / "RESULTS.md") + "'",
              "~~~", "",
              "重建分析命令（产生另一个新报告目录）：", "", "~~~powershell",
              "& 'python' -B .\\p2h_v_decomposition\\analyze_v.py",
              "~~~", ""]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    for p in paths:
        assert digest(p) == hashes[str(p.relative_to(ROOT))]["sha256"], p
    print("ANALYSIS_VERIFIED", out)
    print(json.dumps({c: {a: v[a]["overall"]["contribution_to_all_scene_mean"]
                          for a in ("original_Qwen_vs_MAPPO", "frozen_candidate_vs_MAPPO")}
                      for c, v in analyses.items()}, separators=(",", ":")))


if __name__ == "__main__":
    main()
