# P0: 用 DeepSeek API 独立替代 MAPPO 策略的最小实验

> 对应项目 `code_marl_v0_exper_1`。保留原第五章网络环境、合作博弈、迁移与重路由代价、奖励函数和 MAPPO 训练代码。P0 不是已得到科研优势的算法成果。

## 0. 已经验证 / 尚未验证

**已离线验证**（使用用户上传的 5 个真实 GEANT 场景）：

- `p0_policy.py` 结构化解码原始 240 维观测；4 个 agent 的动作输出维度；比例 `[0,1]` 到环境动作 `[-1,1]` 的转换，及 `agent_g` 第 4 维中性补全。
- JSON 缺字段、额外字段、NaN/Inf、超范围、字符串、维度错误、模型响应截断、API 异常处理，以及不隐式把失败冒充 LLM 成绩。
- 原 `CustomMultiAgentEnv.eval_env_step()` 上的 5 场景闭环：固定、规则、随机、模拟动作和模拟 API，原收益/代价记录逻辑未改。
- `deepcopy` 场景 + 逐场景固定随机种子后，同一场景同一动作得到完全一致的指标，不污染原始样本。
- 修改脚本针对原始 `main.py` 实测：只动样本数与样本文件路径这两处；保留 CRLF 换行，创建备份。

**必须在你的 Conda `marl` 环境内进一步验证**：

- XuanCe **1.2.2** 冻结 MAPPO checkpoint 是否能通过现有代理接口装载和推理；这里缺少 XuanCe 旧版本环境。
- 真正的内网 DeepSeek-v4-flash-0731 HTTP 请求；当前环境无法访问研究室内网。
- 原版完整 600 场景、模型波动和性能统计（目前绝对不能把模拟 API 测试结果写成 LLM 实验结果）。

## 1. 文件放置

把这份压缩包里的文件解压到 **`main.py` 所在项目根目录**：

```
code_marl_v0_exper_1/
  main.py                    # 保留原有，运行补丁器后仅改两处数据加载配置
  configs_main.yaml         # 原有，不改
  utlis/                    # 原有，不改
  data/GEANT/               # 原有完整数据继续留在本地
  P0_small_samples.pkl      # 你已经从 1.8GB 文件导出的小样本
  P0_small_samples.indices.json
  p0_patch_main.py          # 新增
  p0_policy.py              # 新增；不依赖原有 llm_controller.py
  p0_runner.py              # 新增
  p0_compare.py             # 新增
  p0_selftest.py            # 新增
  P0_README.md
```

请使用之前**确实成功复现 MAPPO** 的 `marl` Conda 环境，而非 `base`，也不要直接升级 XuanCe/PyTorch。

## 2. 第一次运行（PowerShell）

```powershell
conda activate marl
cd F:\Projects\code_marl_v0_exper_1

# 第一次仅运行一次。自动备份原 main.py 到 main.py.p0-original-backup
python p0_patch_main.py

# 全部不调用内网 API
python p0_selftest.py
python p0_runner.py --dataset mini --policy mock --count 5 --output-dir p0_runs\mini_mock
python p0_runner.py --dataset mini --policy fixed --count 5 --output-dir p0_runs\mini_fixed
python p0_runner.py --dataset mini --policy rule --count 5 --output-dir p0_runs\mini_rule
python p0_runner.py --dataset mini --policy random --count 5 --output-dir p0_runs\mini_random
```

注意：`mock` 是**模拟动作**，`fixed` 是固定资源比例，不代表 DeepSeek 的性能。

如果 `p0_runner.py` 报 `No module named xuance`，说明没进入原来的 `marl` 环境；先确认 `conda env list` 和 `python -m pip show xuance`。

## 3. 冻结 MAPPO 对照（本地验收关键步骤）

```powershell
python p0_runner.py --dataset mini --policy mappo --count 5 --checkpoint "marl_models\mappo\dir_model_1\512000_GEANT_mappo_model_save.pth" --output-dir p0_runs\mini_mappo
```

此分支**不会训练**，直接创建 XuanCe 代理并将 `.pth` 权重严格载入 `agents.policy`，然后调用原 `Agents.action(..., test_mode=True)` 生成动作。需要确保模型权重是你复现时的 checkpoint。如果实际 checkpoint 在其他路径，直接用 `--checkpoint` 指向真实文件。

如果出现 `AttributeError`、`missing keys` 或 `unexpected keys`，**不要运行原训练程序重训**，把报错发回来：要先核对本地 XuanCe 的具体代理 API / checkpoint 格式。严禁悄悄换成新模型或随机权重。

## 4. 真正的 DeepSeek API P0（本地验收关键步骤）

先设置环境变量，不要把令牌写到 Python 或分享的终端日志里。仅在内网可信环境使用，HTTP 传输未加密。

```powershell
$env:LLM_API_KEY="在这里填你自己的令牌"
$env:LLM_BASE_URL="http://192.168.131.35:30001/v1"
$env:LLM_MODEL="deepseek-v4-flash-0731"

# 先只跑一个真实决策
python p0_runner.py --dataset mini --policy llm --count 1 --fail-mode abort --output-dir p0_runs\mini_llm_1

# 一个成功后再跑全部五个真实决策
python p0_runner.py --dataset mini --policy llm --count 5 --fail-mode abort --output-dir p0_runs\mini_llm_5
```

默认 `--fail-mode abort`：如果 API 错误、超时、输出不是合法 JSON、动作超范围，则立即报错；之前已成功的样本仍写入 JSONL。这样可以避免把规则回退误认为 LLM 成绩。

可选 `--fail-mode rule`：API 错误时执行确定性规则动作，**每个回退都会在记录里标记**，并汇总为 `fallback_rate`。带回退的结果必须单独报告，不能直接当作纯 LLM 结果。

如模型长时间推理可以尝试 `--api-timeout 180 --max-tokens 8192`。这不会更改环境决策逻辑。

## 5. 生成可比的对照表

确保比较中不同方法使用相同的 **原始场景索引**：小样本映射为 `0,1,2,299,599`。

```powershell
python p0_compare.py p0_runs\mini_mappo p0_runs\mini_llm_5 p0_runs\mini_rule p0_runs\mini_random p0_runs\mini_fixed --output p0_runs\mini_comparison.csv
```

`p0_compare.py` 会拒绝场景集合不一致的比较，避免把两批不同样本的均值混为一谈。

每个运行目录有：

- `*_records.jsonl`: 单场景动作、g_done、reward、V/Vk/Vg、代价、推理/执行耗时、原始模型回复（可能涉及算网资源状态，勿公开分享）、是否回退。
- `summary.json`: 对应策略的接纳率、平均效益、平均成本、平均延迟与回退率。
- `manifest.json`: 场景索引映射、代码哈希、模型、seed 和数据路径，不包含 API Key。

## 6. 完整 600 场景数据

原始文件 `data/GEANT/GEANT_3_12_600_sample_save.pkl` 留在你电脑上即可，不需要发给我。大范围实验先试较少样本，然后逐渐扩大：

```powershell
python p0_runner.py --dataset full --policy rule --count 20 --output-dir p0_runs\full_rule_20
python p0_runner.py --dataset full --policy llm --count 20 --fail-mode abort --output-dir p0_runs\full_llm_20
python p0_runner.py --dataset full --policy mappo --count 20 --checkpoint "marl_models\mappo\dir_model_1\512000_GEANT_mappo_model_save.pth" --output-dir p0_runs\full_mappo_20
```

这些命令会读取 1.8 GB 的 pickle 文件，解序列化后内存开销可能显著高于文件体积。不要并行启动多个完整数据任务。

原来的 900 MB `512000_GEANT_env_save.pkl` **不需要上传，也不由新增的 P0 入口读取**。原 `main.py` 的训练/测试路径仍然保留，可在需要时继续使用该环境文件。

## 7. 设计边界和科研限制

- LLM 只看四个 agent 原始 observation 组织出来的结构化信息，没有额外读入 `self.DG` 的隐藏变量。**但它一次获得四个 agent 的联合观察**，而 MAPPO 各 Actor 实际按局部观察分散行动，因此这仍属于集中式 vs 分散式的先导比较；论文正式主对照需要单独标注信息权限差异。
- 原代码的 `agent_g` 动作声明 4 维，但 `_get_g_adjustment()` 只使用前三维，P0 固定第四维为 0（环境映射后 0.5，但未使用）。
- 原环境新切片 CPU/cache 准入分支可能使用调整前的资源需求进行判断，另有 `fixed_tm_idx=True`。**P0 不自行修复**，因为改原环境会破坏和已训练 MAPPO checkpoint 的对照。后续应独立审计，再用同一修订环境重评所有方法。
- `stage_scene()` 每次深拷贝 5 元组并按原始场景索引重置随机种子，以确保不会因执行次序不同污染被比较的场景。
- 这里的样本不是留出网络拓扑集，不能仅凭这批数据证明泛化能力；正式论文需定义独立的留出场景/拓扑及统计区间。
- LLM 未作针对任务的强化学习训练；启发式回退、格式校验都不构成 RL。

## 8. 遇到错误如何反馈

请只发以下任意一种信息：

- `python p0_selftest.py` 的完整异常（若有）；
- `--policy mock` 运行日志；
- `--policy mappo` 运行日志，尤其 checkpoint 加载/actor 接口错误；
- `--policy llm --count 1` 运行日志，**先删除任何可能出现的 token/API Key**。

一旦这两项本地联调都通过，再扩大到多场景评估并考虑下一阶段约束校验机制和论文创新点。
