# P0-v4：Qwen 资源准入硬约束修正器（独立实验，不覆盖原代码）

## 为什么要改变方案

P0-v1/v2/v3a/v3b 只依赖自然语言提示词，无法保证 Qwen 的 15 个有效参数满足原环境的 g 准入要求。v3b 的两个场景仍 `g_done=0`。再增加提示词不能构成保证。

这次新增的是 **Qwen 原始动作 → 数值约束修正 → 原环境原样复验 → 只执行已验证动作**。不修改 `main.py`、`p0_policy.py`、`p0_runner.py`、MAPPO 或奖励函数。只添加三个 `.py` 文件。

**科研归属：**修正后的结果必须标注为 `Qwen + LP admission shield`，而非 `Qwen-only`。优化器读取了真实场景图数据，是高于原先 240 维智能体观测的信息条件。附有 `--solver-only` 对照，以区分 Qwen 候选动作的贡献和数值求解器的贡献。

## 程序做了什么

1. 保留并记录 Qwen 输出的原始动作（15 个有效参数），在全新影子场景中用原来的 `CustomMultiAgentEnv.step()` 测试 `g_done`。
2. 如果原动作已能接纳 g，直接原样保留，完全不修正。
3. 如果失败，按照真实物理节点/链路映射，把容量检查写成线性不等式：
   - 已有切片各节点和链路释放量：原分配容量 × `0.8 × r_k`；同一控制组匹配原环境 `ceil(n/2)` 的动作分组。
   - g 的节点检查：**照原环境实现**，针对缩减前原始计算/缓存请求（与直觉不同）；g 的计算/缓存动作本身不能放宽这个原始准入判断。
   - g 的链路容量：原带宽 × `(1 - 0.7 × r_g_bw)`；已有切片链路释放量仍按 `0.8 × r_k_bw`。
   - g 的链路利用率必须 ≤ 0.7，利用原始流量矩阵和固定最短路径计算。
4. 使用 `scipy.optimize.linprog` 最小化相对 Qwen 候选动作的调整量，优先尝试少改原始决策。
5. 在完全隔离的影子环境中**重新调用原始 `step()` 复验**。不满足 `g_done=True` 或指标非有限值，就继续尝试略大数值裕量；仍未通过就报错，**绝不将失败动作交给正式环境执行**。
6. 单独保存原始/修正后动作、复验结果、执行收益、修正耗时，以及约束缺口诊断。

这是一道**提交前安全门**，并不能断言每个场景数学上都有可行解。如果真的不可行或求解器与真实环境不一致，工具会“失败即拒绝”，而不是伪造接纳成功。

## 解压

把压缩包里的 `p0_guard.py`、`p0_guard_runner.py`、`p0_guard_selftest.py` 放到 Windows 项目根目录（与 `main.py`、`p0_runner.py` 同一层）。**不会覆盖任何旧文件。**

继续使用已成功运行的 `(marl)` Conda 环境、GEANT 数据、`P0_small_samples.pkl` 和索引 JSON。

### 1. 测试依赖及真实场景自检（不用 API）

```powershell
conda activate marl
cd path/to/code_marl_llm_slicing
python -c "import scipy; print(scipy.__version__)"
python p0_guard_selftest.py
```

如果 `scipy` 没安装，先告知其报错，不要盲目更新整个环境。`linprog(method="highs")` 需要 SciPy 1.6 或更新版本。自检使用真实 GEANT 小样本内场景 0、1 的 P0-v3a Qwen 原动作，无任何网络调用。

### 2. 先重放已有的 Qwen v3a 两场景动作（不用 API，推荐）

```powershell
python p0_guard_runner.py --dataset mini --indices 0,1 --raw-records .\p0_runs\qwen_p0_v3a_probe\llm_records.jsonl --output-dir .\p0_runs\p0_v4_replay_v3a
```

正常应显示 `scene=0 raw_admitted=0 verified_final=1 repaired=1` 和 `scene=1 raw_admitted=0 verified_final=1 repaired=1`。回放的 `api_latency_s` 是 0（没有真的调用 API），不是 Qwen 的原响应时间；真实费用和时延要看现场调用记录。

如果想测试 v3b 已生成的日志，也可以把 `--raw-records` 换成 `qwen_p0_v3b_probe\llm_records.jsonl`。这里以真实日志为准，不要手抄数字。

### 3. 直接调用 Qwen 并开启硬约束安全门

```powershell
python p0_guard_runner.py --dataset mini --indices 0,1 --model qwen3-next-80b-a3b-instruct --prompt-version p0_v3b --api-timeout 20 --max-tokens 2048 --output-dir .\p0_runs\p0_v4_qwen_live
```

流式请求默认启用。此命令只对结果通过原环境复验后执行正式环境的 `step()`。未设置 `LLM_API_KEY` 会停止并提示配置；不会自动使用规则策略代替 Qwen。

当完成 0、1 场景测试后，可以使用 `--indices 0,1,2,3,4` 测试五个小样本（对应原场景 0、1、2、299、599）。每轮用新的 `--output-dir`，防止覆盖旧结果。

### 4. 必做的科学对照：不使用 Qwen，只运行相同的约束求解器

```powershell
python p0_guard_runner.py --solver-only --dataset mini --indices 0,1 --output-dir .\p0_runs\p0_v4_lp_only
```

该策略输出标记 `lp_only_baseline`，`source_model` 为 `null`，不调用 LLM。用于检验：P0-v4 的改进来自 Qwen 的候选提议，还是仅由 LP 修正器驱动。

## 输出与评价

- `guard_records.jsonl`：每个场景记录 `raw_action`（原 Qwen 决策）、`action`（最终提交）、`raw_check`、`guard_check`、`guard_changed`、`guard_details`、`g_done`、`V`、`V_g`、`cost_sum`、三个耗时字段。`source_model`、`prompt_version`、`policy` 可区分策略。
- `summary.json`：修正后接纳率 `acceptance_rate`、修正前提议可行率 `qwen_raw_acceptance_rate`、修正比例 `repair_rate`、平均 V/成本/时间。
- `manifest.json`：模式（真实 Qwen / 日志回放 / 纯 LP 对照）、模型信息、样本、随机种子、源文件摘要。

如果修正器报 `GUARD FAIL-CLOSED`，请保留日志并上报。该场景不会进入原正式 `step()`，不能计入“已接纳”，也不能删除后只报告成功场景；后续开展 600 场景的正式统计前，要明确定义无解/拒绝场景如何计入分母。

## 已完成的离线验证

在独立 Python 运行环境中，通过对 XuanCe/Gym 启动接口进行最小模拟，实际加载了上传的 `main.py` 环境业务代码和真实 5 场景数据，进行了真实 `env.step()` 重放。

- P0-v3a 场景 0：Qwen `g_done=False` → 修正后 `g_done=True`，`V≈283.906`。
- P0-v3a 场景 1：Qwen `g_done=False` → 修正后 `g_done=True`，`V≈337.260`。
- P0-v1 原始 Qwen 五场景记录：五次修正均通过了原环境复验（这些是**修正器**成绩，不是 Qwen 独立成绩）。
- MAPPO 五个已接纳动作：原样保留，均未被修正。
- 单元测试涵盖：真实环境重放、影子场景不污染正式图、已可行动作不修改、数学无解时拒绝、求解与原环境结果不一致时拒绝、拒绝回放规则兜底输出。

**仍待你本地核验：**在你的 `(marl)` XuanCe 1.2.2 环境、真实 Qwen 网关上运行上述自检和 `live` 命令。这里只完成了原环境核心逻辑与真实样本的离线测试，未直接连接内网 API。
