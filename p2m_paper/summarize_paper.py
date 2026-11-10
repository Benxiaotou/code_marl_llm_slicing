"""Summarize completed supplements; never constructs a policy or API client."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'p2m_paper/cached_20261105T025855_555782Z'
NEW=ROOT/'p2m_paper/fresh_20261105T025939_435887Z'


def read(path,cap=500000):
    assert path.is_file() and path.stat().st_size<cap
    return json.loads(path.read_text(encoding='utf8'))


def main():
    a=read(OLD/'summary.json'); b=read(NEW/'summary.json')
    assert a['status']==b['status']=='COMPLETE'
    assert read(OLD/'audit.json')['passed'] and read(NEW/'audit.json')['passed']
    api_path=NEW/'api_stats.json'; report=ROOT/'p2m_paper/PAPER_SUPPLEMENT_REPORT.md'
    assert not api_path.exists() and not report.exists()
    source=NEW/'responses.jsonl'; assert source.is_file() and source.stat().st_size<10000000
    responses=[json.loads(line) for line in source.open(encoding='utf8')]
    assert len(responses)==b['API_calls']==6144
    lat=np.array([r['latency_s'] for r in responses],dtype=float)
    api={'successful_requests':len(responses),'model':responses[0]['model'],'GPT_calls':0,
         'estimated_additional_fee_usd':0,'cost_basis':'User unlimited internalQwen quota',
         'requests_without_usage':sum(not r['usage'] for r in responses),
         'request_latency_s':{'mean':float(lat.mean()),'median':float(np.median(lat)),'p95':float(np.quantile(lat,.95))},
         'latency_definition':'Client-measured single-role HTTP completion time;not full-scene or total experiment runtime.',
         'tokens':b['token_usage'],
         'inputs_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (OLD/'summary.json',NEW/'summary.json',source,Path(__file__).resolve())}}
    with api_path.open('x',encoding='utf8') as f:
        json.dump(api,f,indent=2)
    ranked=sorted(b['summaries'],key=lambda k:b['summaries'][k]['V_mean'],reverse=True)
    text=['# 论文补充实验汇总','',
          '两轮均已完成并通过审计：旧测试1536场归因消融；三个预先锁定新种子各512场的独立确认。',
          '每轮MAPPO、kNN、Qwen分别有原始/仅链路/仅节点/链路+节点四个版本。参数固定，无调参或可选停止。','',
          '| 方法 | 旧1536平均V | 新1536平均V | 新接纳率 | 新合作有效率 | 新平均成本 |',
          '|---|---:|---:|---:|---:|---:|']
    for name,s in b['summaries'].items():
        text.append(f"| {name} | {a['summaries'][name]['V_mean']:.4f} | {s['V_mean']:.4f} | "
                    f"{s['acceptance_rate']:.2%} | {s['cooperative_rate']:.2%} | {s['cost_mean']:.4f} |")
    text+=['','## 新场景预声明V主张','','| 比较 | ΔV | 配对95%区间 | Bonferroni三项主张区间 |','|---|---:|---|---|']
    for name in ('candidate_vs_MAPPO','matched_MAPPO','matched_kNN'):
        c=b['comparisons'][name]['V']; lo,hi=c['CI95']; blo,bhi=c['Bonferroni_family3_CI98_333333']
        text.append(f"| {name} | {c['mean_delta']:+.4f} | [{lo:+.4f},{hi:+.4f}] | [{blo:+.4f},{bhi:+.4f}] |")
    text+=['','## 解读','',
           '新场景平均V最高的版本为'+ranked[0]+'；原固定Qwen_both按均值排第'+str(ranked.index('Qwen_both')+1)+'。',
           '完整比较必须保留校正后的MAPPO与kNN；相对未校正MAPPO的提升不能全部归因于LLM。']
    supported=all(b['comparisons'][k]['V']['Bonferroni_family3_CI98_333333'][0]>0
                  for k in ('candidate_vs_MAPPO','matched_MAPPO','matched_kNN'))
    text.append('三个预声明V比较在家族校正后均为正。' if supported else
                '当前结果未同时支持三个预声明V优势；不能声称LLM组合在原MAPPO及相同校正基线上全面具有V优势。')
    text+=['接纳率、合作有效率、成本及各参与者效益应同时呈现；更多接纳不自动意味着更多满足个体理性的合作。',
           'MAPPO_node等非主比较版本的均值排名是描述性结果，不把事后最高版本替换为原预声明主要候选。',
           '旧测试此前已查看，仅用于归因分析；新测试方案在旧消融结果出现前锁定且全部12版本完成。',
           '不把两轮或以前验证集混合成更有利的主要结果。新测试不能再用于调参后声称未使用最终测试。',
           '同GEANT生成器下的独立样本不证明跨拓扑泛化；每场一次Qwen抽样不证明重复推理稳定性。',
           '原kNN取最近单案例，P2-A Qwen另有IR重排和多案例提示，二者差值不全是纯LLM推理效应。',
           '没有智能体间通信；V为原计账效益，不是端到端吞吐量。','',
           '## 完整性与API','',
           '4项契约测试通过；旧阶段18432条指标/动作审计、1536原kNN控制、24历史探针、36后控制通过。',
           '新阶段18432条指标/动作、6144提示词/响应、1536 MAPPO复现、36后控制及源文件/数据哈希审计通过。',
           '6144次内网Qwen请求，无失败/重试/回退；GPT调用0；新增API费用估计$0（用户无限内网额度）。',
           'usage缺失请求数：'+str(api['requests_without_usage'])+'。缺失token用量未知，不能报告为0。',
           f"单角色请求延迟：均值{lat.mean():.3f}s，中位数{np.median(lat):.3f}s，p95{np.quantile(lat,.95):.3f}s。",
           '上述延迟不是完整场景决策耗时，也不是整个实验运行时长。内网模型部署权重版本及计费条件未独立核验。','',
           '新建代码、图表、报告均在p2m_paper；PROJECT_HANDOFF.md更新。原环境、基线、数据、检查点和指标未修改。',
           '检查文件与实际测试/复现命令详见README.md；两轮逐种子与完整消融结果各见对应RESULTS.md。','',
           '查看本报告（无API）：','','```powershell',
           'Get-Content -LiteralPath .\\p2m_paper\\PAPER_SUPPLEMENT_REPORT.md','```','']
    with report.open('x',encoding='utf8') as f:
        f.write('\n'.join(text))
    print('PAPER_REPORT_CREATED',report)
    print(json.dumps({'ranking':ranked,'API':api['request_latency_s'],'missing_usage':api['requests_without_usage']}))


if __name__=='__main__':
    main()
