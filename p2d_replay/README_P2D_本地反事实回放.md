# P2-D｜冻结 MAPPO × GPT-6.1 Sol：10 场景本地反事实回放

## 目的及当前限制

使用 **seed7601 的场景 0–9**，读取你已保存的两组动作，重新交给你本机原始 GEANT 环境的 `env.step` 评分。**不重新请求 GPT/Qwen，不重新生成场景，不改 Prompt、环境代码、训练模型或 P2-C 测试数据。**

**此压缩包是可执行脚本，不是已经算出的反事实结果。** ChatGPT 所在环境只收到了记录 ZIP，没有你 Windows 本机的 `main.py`、GEANT 数据、场景 `.pkl` 和 MAPPO checkpoint，因此不能在这里完成真正的 GEANT `env.step`；请在你现有的 `marl` 环境运行。我们已经对上传的诊断 ZIP 做了预检，并运行了离线单元测试。

| 组合代号 | k1/k2/k3 动作来源 | g 动作来源 | 意义 |
|---|---|---|---|
| **MM** | MAPPO | MAPPO | 原结果重放控制组 |
| **GG** | GPT | GPT | 原结果重放控制组 |
| **MG** | MAPPO | GPT | 保留 GPT-g，替换原有切片的联合动作 |
| **GM** | GPT | MAPPO | 保留 GPT-k，替换新切片 g 的动作 |

特别注意：诊断记录中的 GPT `agent_g` 环境动作第四维固定为 0，而 MAPPO 第四维随场景变化。这些动作按**完整原始四维向量**进行交叉，不擅自修复/填充值。不能在确认第四维实际环境语义前把原因归结于某一角色。

## 第一步：把程序解压到项目目录

假设项目目录为：`path/to/code_marl_llm_slicing`。解压后复制 `p2d_counterfactual_replay.py` 到项目根目录。你以前生成的 `GPT61_MAPPO_10scene_diagnosis.zip` 也应位于同一目录。**不需要手工解压诊断 ZIP**；其中包含三个重名 `records.jsonl`，普通解压可能覆盖数据，本脚本已单独处理。

```powershell
conda activate marl
cd path/to/code_marl_llm_slicing
Test-Path .\GPT61_MAPPO_10scene_diagnosis.zip
Test-Path .\p1c_runner.py
Test-Path .\p1b_data\gpt61_dev_seed7601.json
Test-Path .\marl_models\mappo\dir_model_1\512000_GEANT_mappo_model_save.pth
```

以上都应为 `True`。不要把诊断 ZIP 换成 P2-C 正式测试记录。

## 第二步：零成本预检（不加载 GEANT）

```powershell
python .\p2d_counterfactual_replay.py --preflight
```

预检应看到 `PREFLIGHT PASS`、`mappo_scenes: 10`、`gpt_scenes: 10`、`recorded_mappo_accept_count: 6`、`recorded_gpt_accept_count: 0`。这只是档案一致性校验，**不等于已完成环境回放**。

## 第三步：执行四组动作回放（零 API 费用）

```powershell
python .\p2d_counterfactual_replay.py `
  --project-root . `
  --archive .\GPT61_MAPPO_10scene_diagnosis.zip `
  --dataset-manifest p1b_data\gpt61_dev_seed7601.json `
  --library p2b_libraries\p2b_standard_1000_v2 `
  --checkpoint marl_models\mappo\dir_model_1\512000_GEANT_mappo_model_save.pth `
  --seed 1 `
  --output-dir gpt61_runs\dev7601_cross_replay_v1
```

脚本会检查数据 manifest 中的样本 SHA-256、环境 `main.py` 与配置版本，继而使用你项目中已经跑通的 `p1c_runner.create_eval_env(...)` 与 `p1a_runner.run_one_scene(...)`。为了尽可能复用原环境评分流程，它创建冻结 MAPPO 教师策略以进入**原有运行器**，但在真正调用 `env.step` 的唯一位置，将联合动作替换为先前记录的 **缓存动作**。**最终评分不是教师重新推理出的新动作**，而是所指定 MM/GG/MG/GM 组合的历史动作。运行时不构造 LLM API 客户端。

**重要：**只有两组原样回放的控制组（MM/GG）都在 10 个场景逐项复现已有 `V`、`V_g`、成本、IR 和接纳判定（默认 `1e-4` 绝对容差），才会跑 MG、GM；否则立即终止并保存 `CONTROL_FAILED.json`，禁止把交叉结果用于结论。出现控制不一致时，不要放宽容差来强行过关。

首次成功后**不要重复执行并覆盖**，脚本拒绝覆盖非空输出目录。

## 第四步：读取结果

```powershell
Get-Content .\gpt61_runs\dev7601_cross_replay_v1\summary.json -Raw
Import-Csv .\gpt61_runs\dev7601_cross_replay_v1\paired_crossover_by_scene.csv |
  Format-Table -AutoSize
```

输出文件：

- `protocol.json`：数据/代码/checkpoint 哈希、交叉动作来源、样本列表。
- `MM_records.jsonl`、`GG_records.jsonl`、`MG_records.jsonl`、`GM_records.jsonl`：原环境评分结果和真实送入 `env.step` 的动作；**不会复制历史 Prompt**。
- `summary.json`：四组的平均接纳率、合作有效率、效益 V、新切片效用 V_g、调整成本。
- `paired_crossover_by_scene.csv`：十个场景逐项并列对照。
- `scene_metrics.csv`：各条件每场景细节、三个已有切片的 IR 余量。
- `CONTROL_FAILED.json`：仅在原样回放校验失败时出现。

把 **`summary.json` + `paired_crossover_by_scene.csv`** 发给我即可。无需上传 API Key、GPT 原始 Prompt 或整个源代码。

## 如何解释（先看实际结果，不预设答案）

- 若 **MG 明显恢复接纳、GM 仍失败**：支持「已有切片 k1/k2/k3 的资源调整」是更重要的失败因素。
- 若 **GM 明显恢复、MG 仍失败**：支持「新切片 g 的动作／四维映射」是更重要的因素。
- 若 **MG 和 GM 都失败**：可能存在强烈的联合动作协同依赖，不能单凭单侧替换归因。
- 若 **MG/GM 均改善**：两侧都可能存在可修复的动作偏差，需要结合 IR 余量分析。

这些结论只适用于这十个开发场景的**确定性动作干预**，不能将事后交叉动作分数当作独立可部署策略的性能，也不能用来调 P2-C 正式测试集。

## 出错排查

- `Archive missing`：确认 ZIP 在项目根目录且名称正确；也可以用 `--archive` 传绝对路径。
- `Original environment/config changed since dataset generation`：先核对生成 seed7601 时的 `main.py`/配置是否已经变动，不要绕过哈希验证。
- `CONTROL MISMATCH ...`：本机原环境与历史记录不能复现，交叉结果无效；先分享 `CONTROL_FAILED.json` 和报错。
- `Environment action dict roles ...` 或 `Unknown joint action structure`：运行器的 `env.step` 入参格式与程序假设不符。不要猜动作维度，提供堆栈中相关的 `p1a_runner.run_one_scene` 调用片段进一步适配。
- 任何输出目录覆盖错误：这是为了保护原始归档，请先检查目录内容，或改用新输出目录名。

**8 项纯离线检查已通过；由于未取得本机 GEANT/场景文件，尚未执行真实四组环境反事实评分。**
