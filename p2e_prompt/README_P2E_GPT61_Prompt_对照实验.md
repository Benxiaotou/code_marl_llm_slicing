# P2-E · GPT-6.1 Sol 三版 Prompt 对照实验（seed7601 / 10 场景）

**用途：** 复用你已运行过的同一份新开发数据集 `gpt61_dev_seed7601` 的 **scene 0～9**。不重新生成场景，不修改 MAPPO、原版 Qwen、`p1c_runner.py`、`main.py`、历史经验库或 P2-C 正式测试结果。每版 Prompt 运行 10 个配对场景，用原有 GPT Prompt 已经完成的 10 个场景作为免费的历史基线。

## 设计

| 组别 | 目的 | 与原版 Prompt 的差异 |
|---|---|---|
| GPT_original | 原版 GPT `IR-aware + Standard-1000` | 从已保存的 `gpt61_runs/dev7601_iraware_std1000_v1` 读取，不再次调用 API |
| semantic_balance | 约束语义澄清版 | 明确“低节点释放不等于高 IR”，同时避免过度释放，突出节点和链路不能互相替代 |
| teacher_scale | 历史动作尺度校准版 | 从自己的三条 MAPPO 案例获取大致尺度并按局部观测调整，而非盲目压低节点控制 |
| joint_feasibility | 联合资源可行性版 | 用“动作语义→本地节点重叠→IR/成本→节点链路一致性”的静默决策清单 |
| MAPPO | 固定参照 | 直接读取此前冻结 MAPPO 在同 10 场景上的结果 |

**重要控制：** 三个变体仅在 `agent_k1/k2/k3` 的原始系统 Prompt 结尾追加一个短段。`agent_g` 的原 Prompt 完全不变；局部观测、三条检索示例和训练期 IR 反馈不变。模型仍用 `gpt-6.1-sol`、`medium`、IR-aware、Standard-1000、`seed=1`，每版完整调用四角色 10×4=40 次。`g` 仍会重新调用：提示词相同并不意味着随机生成结果严格相同。

> Prompt 的三版内容分别位于 `prompts/01_*.txt`、`02_*.txt` 和 `03_*.txt`，均可直接打开查看。不要在同一次实验中继续修改它们：程序会记录 SHA-256 并锁定实验配置。

## 运行成本

此前你实测：GPT 10 场景 40 请求，输入 **215,978 Token**、输出 **4,502 Token**（其中推理 **3,502 Token**，已包含在输出 Token 内）、预估 **$0.4770**。

沿用相同计费规则，三版各 10 场景约 **$0.48/版，总计约 $1.43**，实际值会随 Prompt 长度、推理 Token、缓存和网关计费变化。每版默认 **$2 的程序内估算预算保护、最多 40 条 API 请求**，因此三版总体最多按三个独立账本控制；它不能替代网关侧的真实费用上限（网关加价、失败请求计费等可能不一致）。每版采用两个 5 场景分段；完成后可跳过已验证分段。

## PowerShell 执行步骤

在 Windows 的 `marl` 环境中，先将 ZIP 放到项目根目录。

```powershell
conda activate marl
cd path/to/code_marl_llm_slicing
Expand-Archive .\P2E_GPT61_Prompt_3way_v1.zip -DestinationPath .\p2e_prompt -Force
```

沿用你上次配置好的网关端点和密钥。如果在新 PowerShell 会话里，需重新设置环境变量：

```powershell
$env:GPT61_BASE_URL = "https://你的实际端点/v1"
$secureKey = Read-Host "请输入 API Key" -AsSecureString
$env:GPT61_API_KEY = [System.Net.NetworkCredential]::new("", $secureKey).Password
Remove-Variable secureKey
```

密钥**不要**发到聊天、不要放进脚本、不要提交 GitHub。

### 1. 离线检查（不花钱）

```powershell
python .\p2e_prompt\p2e_prompt_3way.py --mode check --project .
```

应看到 `DATASET VERIFIED`，以及三版的 `LOCK PREVIEW`。检查失败时请先停止，尤其是数据集哈希不一致。

### 2. 建议先跑第一版，10 场景

```powershell
python .\p2e_prompt\p2e_prompt_3way.py --mode run --project . `
  --variant semantic_balance --budget-usd 2
```

每版会依次跑 scene 0～4 与 5～9；完成输出各自的 `summary.json` 和 `usage.jsonl`。

### 3. 第一版结束后，可以先看看和原版有无区别（零费用）

```powershell
python .\p2e_prompt\p2e_prompt_3way.py --mode compare --project .
Import-Csv .\gpt61_runs\dev7601_prompt_ablation_v1\comparison.csv | Format-Table -AutoSize
```

### 4. 视第一版结果决定是否继续另外两版

```powershell
python .\p2e_prompt\p2e_prompt_3way.py --mode run --project . `
  --variant teacher_scale --budget-usd 2

python .\p2e_prompt\p2e_prompt_3way.py --mode run --project . `
  --variant joint_feasibility --budget-usd 2

python .\p2e_prompt\p2e_prompt_3way.py --mode compare --project .
```

若你的网关只支持 Responses API，可在**首次运行之前**加 `--api responses`，但同一组已经开始后不能切换协议。若你此前确实使用非官方 Token 单价，也应在首次运行前通过 `--input-usd-per-m ... --output-usd-per-m ... --cached-usd-per-m ...` 指定，之后保持不变。

## 输出与核验

每版结果目录：

`gpt61_runs/dev7601_prompt_ablation_v1/<variant>/`

包含：
- `protocol_lock.json`：配置、源文件及 Prompt SHA-256 防误改锁定。
- `usage.jsonl`：40 次真实 Token 用量/估算费用记录。
- `prompt_injection_audit.jsonl`：**30 次**已有切片 Prompt 注入 + **10 次**新切片 g 不变的 SHA-256 核查日志；不记录原始密钥。
- `chunks/0000_005` 与 `chunks/0005_005`：每 5 场景的原始 `records.jsonl`、`metrics.csv`、`summary.json`。
- `summary.json`：该版合计 10 场景的接纳、合作有效、V、V_g、成本、时延和费用。

运行 `--mode compare` 后，统一生成：
- `gpt61_runs/dev7601_prompt_ablation_v1/comparison.csv`
- `gpt61_runs/dev7601_prompt_ablation_v1/comparison.json`
- `gpt61_runs/dev7601_prompt_ablation_v1/paired_10scene_metrics.csv`

**注意：** 原版 GPT 已经 0/10 接纳，MAPPO 为 6/10。看新 Prompt 是否恢复接纳时，还必须同步看合作有效、IR、系统效益 V、新切片效用 V_g、调整成本、节点控制动作；不能只看接纳率。三版 Prompt 均在这十个开发场景上筛选，任何看起来最优的版本都必须再用**新的开发/独立测试场景**验证；不能声称已经超过 MAPPO。

## 失败处理

若发生请求超时、网关拒绝、预算保护或非 JSON 输出，程序会保留已写入的目录及 API 账本；它**不会静默覆盖**中断的五场景分段。请保留日志，不要直接删除目录后重跑，以免重复计费。

离线自检（不调用 API）：

```powershell
python -m unittest discover -s .\p2e_prompt\tests -p test_prompt_3way.py -v
```

如果只想让我分析结果，上传 `comparison.csv`、`paired_10scene_metrics.csv` 和各版 `summary.json` 即可，不需要提供密钥或端点。
