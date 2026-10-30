# P1-A+：局部资源语义增强的 Qwen 分布式决策实验

**实验目的：**与原 P1-A 的四智能体 Zero-shot-Qwen 比较，仅增加“由该角色原始局部观测确定性推导的动作含义和静态资源风险信息”，判断这种解释是否改善网络切片接纳、个体理性、效益与成本。**不微调、不加入历史经验、不共享其他智能体当前状态、不用 Guard/LP 修正、不做测试时环境模拟。**

## 0. 解压方式（不会覆盖原有 P1-A）

将本 ZIP 的内容解压到现有项目根目录，即包含 `p1a_runner.py`、`p1a/` 和 `main.py` 的位置：

```
F:\Projects\code_marl_v0_exper_1
    main.py                       (原文件，保持不变)
    p1a_runner.py                (原文件，保持不变)
    p1a/                         (原目录，保持不变)
    p1aplus_runner.py            (本补丁新增)
    p1aplus_compare.py           (本补丁新增)
    p1aplus/                     (本补丁新增)
    tests/test_p1aplus.py        (本补丁新增)
```

原始 P0/MAPPO、原始 `p1a_runner.py`、`p1a/policy.py`、`main.py`、`configs_main.yaml` 和数据文件**均不做修改**。

推荐继续使用你已经验证可工作的 `(marl)` Conda 环境：

```powershell
conda activate marl
cd F:\Projects\code_marl_v0_exper_1
python -m unittest discover -s tests -p "test_*.py" -v
```

本补丁离线已通过 18 项新增测试、原 P1-A 14 项回归测试；也针对之前上传的 5 个真实场景中的 20 份局部提示内容进行了离线结构核对。**尚未在研究室内网运行 Qwen；真实环境联调仍需本地执行。**

## 1. 新语义层提供了什么？

保持原始 240 维局部数值观测及 P1-A 系统提示不变，只追加相应角色的可计算解释：

**k1/k2/k3：**

- 显示两个节点和两个链路控制值分别对应哪些**按原环境局部索引排序**的重叠节点/链路。原代码使用 `np.repeat(...,ceil(n/2))` 连续重复，不是交替分配。
- 明确已有切片的容量变化是 `C_after = C_before * (1 - 0.8*r)`。节点控制值同时作用于计算和缓存容量。
- 对节点控制组，根据观察到的计算/缓存利用率得到“不考虑迁移、当前使用量不变”的**静态风险参考** `min((1-u_comp)/0.8,(1-u_cache)/0.8)`，并说明这不是真正的最优让渡值，也不是硬约束。
- 枚举若干 `r` 取值的剩余容量比例和当前容量不足的节点数量；从未实际执行 `env.step` 或预测迁移成本。
- 链路利用率在原环境可能显示为 0，因此**不把零值直接解释为可以安全让出全部带宽**；重路由损失和最终流量承载能力均是未知的。

**g：**

- 明确计算、缓存、带宽的压缩作用：`C_after = C_before * (1 - 0.7*r)`。
- 根据 g 自身已观察到的物理资源占用与请求量，报告**其他切片采取动作之前**的静态缺口。
- 对带宽，计算当前快照下静态满足容量条件所需的压缩率（可能大于 1，即仅靠压缩不足）。这不是对联合动作结果的预测。
- 如实指出原始 `main.py` 中 g 计算/缓存准入条件仍使用压缩前请求量，带宽则使用压缩后请求量。**本补丁保留这个旧环境行为，不暗中修复，需与 MAPPO 保持同一环境。**

> **信息边界：**Prompt 只接受 `observations[agent]` 和公开的原环境规则，不接受 `env.DG`、当前其他角色观测、物理网络完整状态、MAPPO 历史动作或联动后环境奖励。

## 2. 一次本地 Smoke Test（不用 API）

```powershell
python p1aplus_runner.py --dataset mini --policy mock --count 1
```

它会生成 `p1aplus_runs/mock_mini_.../records.jsonl`，里面每个角色的 `messages` 都包含“LOCAL ACTION SEMANTICS”。**Mock 不能视为 Qwen 效果。**

## 3. 先跑 10 个场景，验证真实 API

要求你的原 600 样本文件仍位于 `data/GEANT/GEANT_3_12_600_sample_save.pkl`，你已经跑通的 P1-A 程序可使用 `--dataset full`。

PowerShell 设置当前终端的 API key（不要上传或提交密钥）：

```powershell
$env:LLM_API_KEY = "你的真实令牌"
$env:LLM_BASE_URL = "http://192.168.131.35:30001/v1"
```

先在相同场景、相同模型配置下跑原始 P1-A 和语义增强 P1-A+：

```powershell
python p1a_runner.py --dataset full --policy qwen --start 512 --count 10 --model qwen3-next-80b-a3b-instruct --temperature 0.1 --seed 1 --max-tokens 4096 --api-timeout 120 --on-error abort --output-dir .\p1a_runs\base_dev10

python p1aplus_runner.py --dataset full --policy qwen --start 512 --count 10 --model qwen3-next-80b-a3b-instruct --temperature 0.1 --seed 1 --max-tokens 4096 --api-timeout 120 --on-error abort --output-dir .\p1aplus_runs\semantic_dev10

python p1aplus_compare.py --baseline .\p1a_runs\base_dev10 --semantic .\p1aplus_runs\semantic_dev10 --output-dir .\p1aplus_runs\compare_dev10
```

该命令会完整加载原始 1.8GB 场景数据；需要本地有足够内存。`--api-timeout` 是网络读取超时，不是严格总时间限制。如原代码中动作规则标记与当前 `main.py` 不一致，新运行器会拒绝执行，避免产生错误的语义解释。

**重要：**这些是开发场景，不允许把 10 个结果当成最终结论。10 场景正常后才跑完整开发对照。

## 4. 较大开发对照：80 个原始场景 × 2 方法

由于此前 P0 已分析过原始索引 `0..511`，且 `599` 已用于 P1-A 开发，我们选用 **`512..591` 共 80 个开发场景**。这部分不再作为论文最终未触碰的测试集。若本地样本生成逻辑已经改变，应重新核定数据隔离。

```powershell
python p1a_runner.py --dataset full --policy qwen --start 512 --count 80 --model qwen3-next-80b-a3b-instruct --temperature 0.1 --seed 1 --max-tokens 4096 --api-timeout 120 --on-error abort --output-dir .\p1a_runs\base_dev80

python p1aplus_runner.py --dataset full --policy qwen --start 512 --count 80 --model qwen3-next-80b-a3b-instruct --temperature 0.1 --seed 1 --max-tokens 4096 --api-timeout 120 --on-error abort --output-dir .\p1aplus_runs\semantic_dev80

python p1aplus_compare.py --baseline .\p1a_runs\base_dev80 --semantic .\p1aplus_runs\semantic_dev80 --output-dir .\p1aplus_runs\compare_dev80
```

两种策略各自处理 80 场景、每场景 4 次独立模型请求，合计 **640 次 Qwen 调用**。与 P1-A 相同，四个角色按顺序调用；测量的是顺序调用总时延，不是并行的理论下界。原环境的随机数在执行 `env.step` 前使用相同的 `seed + scene_id` 重置，但模型输出仍可能存在采样/服务端非确定性。

当任何一条策略因无效 JSON、网络/API 异常失败时，默认 `abort` 保存已经记录的结果并终止，不会替换成规则或优化器动作。配对比较工具拒绝丢场景、失败、回退、模型不同、环境哈希不同和其他不匹配配置的运行结果。若某次运行目录存在，换一个新名字；不要覆盖历史实验记录。

## 5. 指标与结果文件

两类运行分别输出：`manifest.json`、`records.jsonl`、`metrics.csv`、`summary.json`。

比较程序输出 `comparison.json` 和 `paired_scenes.csv`，重点字段：

| 字段 | 解释 |
|---|---|
| `admission` | 成功接纳比例 |
| `cooperative` | 接纳且满足个体理性的比例 |
| `V` | 每场景系统效益的配对均值 |
| `V_g` | 新切片效益 |
| `cost_sum` | 已有切片调整成本 |
| `policy_latency_s` | 四个模型调用的实际总时间 |
| `mean_abs_action_delta` | 同场景 15 个动作参数的平均绝对变化 |
| `action_diversity` | 每个角色输出动作的跨场景标准差、唯一动作数 |
| `paired_bootstrap_95pct_interval` | 场景重采样得到的配对均值差区间（开发分析描述性） |

只有当两种策略在所有选定场景都有有效记录时，比较才成功。开发样本的置信区间不能取代独立最终测试。如果后续对测试结果进行 Prompt 调整、选择模型，需重新生成新的严格独立测试场景。

## 6. 怎么判定是否值得继续？

优先观察是否提高 **cooperative_valid（接纳且个体理性）**，然后观察接纳率、总 V、成本、延迟。还要看看同一模型的动作是否真正响应不同局部压力，而不是仍输出固定的 `[0.2, 0.2, ...]`。

如果 80 个开发场景改善有希望，建议固定方案，再新生成未参与调试的独立测试集，至少 256 场景，条件允许可取 512；并对少数场景进行重复调用，检查 LLM 采样波动。不能因为多了语义解释就声称系统效益必然提高。

如果改善很小，下一阶段再考虑任务经验检索（P1-B/C）或换推理模型，不把模型切换和提示词增强同时改变，否则难以判断增益来源。

## 7. 当前限制与安全说明

- 没有真实内网 API 访问权限，因此本补丁只完成离线测试与已有场景记录解析，**没有生成新的 Qwen 性能数字**。
- 原主环境的 g 计算/缓存准入判断可能与预期的压缩机制不一致，必须先明确修复方案并为 MAPPO、P1-A、P1-A+ 同时重跑，不能只修一边。
- `main.py` 检查仅验证源码中关键标记存在，不等于形式化证明全部逻辑正确。如果你本地有改过核心资源逻辑，请先核对该环境版本。
- 回滚只需删除 `p1aplus_runner.py`、`p1aplus_compare.py`、`p1aplus/` 和新增的 `tests/test_p1aplus.py`；原始 MAPPO、P0、P1-A 仍保留。
