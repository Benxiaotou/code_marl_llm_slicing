# GPT-6.1 Sol：10–20 场景接口与费用验证（独立适配包 v1）

## 目的、边界与已有项目

本包只验证“冻结 GPT-6.1 Sol + IR-aware + Standard-1000 经验库”与现有 GEANT/P1-C 运行器的接口、动作合法性、成功率、时延和 Token 消耗；**不使用、更不会改动已定稿的 P2-C seed6601/6701/6801 测试集**。这是一组新开发场景，默认 seed7601、20 个场景；正式跨模型对照仍应使用新开发集及之后独立保留的测试集。

目标项目根目录：`path/to/code_marl_llm_slicing`。复制本包的两个主文件 `gpt61_sol_dev.py`、`run_gpt61_sol_dev.ps1` 到项目根目录；可以连同 `gpt61_adapter_tests/` 和本说明一起复制。**不要覆盖** `p1c_runner.py`、`main.py`、`p2c_eval_runner.py`、经验库、MAPPO checkpoint 或任何已有结果目录。

程序以 127.0.0.1 本机代理适配现有 `p1c_runner.py` 的 Chat Completions 请求：删除旧 Qwen 请求中的 `temperature`/`max_tokens`，替换为 GPT 适用的 `reasoning_effort` 与 `max_completion_tokens`（Responses API 改为 `reasoning` 与 `max_output_tokens`）；最后回传原运行器能识别的 Chat Completions 结构。**不修改现有角色 Prompt、检索方式、动作含义、环境评价函数**。原运行器仍会在自己的实验 manifest 中保留旧式 `temperature=0.1` 字段，但适配器审计记录明确注明**实际发往 GPT 的请求没有 temperature 参数**。

> 已进行 Python 语法和 8 项离线模拟回归测试，**未在你的真实端点、MAPPO checkpoint 或 GEANT 环境上执行**。若网关采用非标准模型 ID、字段或身份验证方式，请先停在 1 次 `probe`，不要直接跑 20 场景。

## 0. 设置 API 端点与密钥（当前 PowerShell 进程）

```powershell
conda activate marl
cd path/to/code_marl_llm_slicing

$env:GPT61_BASE_URL = 'https://你的服务商域名/v1'
$secureKey = Read-Host '输入 API 密钥（输入不回显）' -AsSecureString
$env:GPT61_API_KEY = [System.Net.NetworkCredential]::new('', $secureKey).Password
Remove-Variable secureKey
```

这里的 URL 是示例，请替换为**你的真实网关 API 基础 URL**，不要把模型网页地址误当 API 基础 URL。**不要把密钥粘贴到聊天、代码、日志、GitHub 或截图中**。程序不会将 API 密钥写入磁盘，运行完成后可以用 `Remove-Item Env:GPT61_API_KEY` 清理会话变量。

如果服务商使用仅内部网络的 `http://` 端点，必须明确增加 `-AllowHttpUpstream`，因为 HTTP 无法保护密钥传输。如果网关不兼容 Chat Completions，可在 `probe` 和 `run` 中都传 `-UpstreamApi responses`；请使用相同的接口和模型配置完成整组实验。

## 1. 只读查看配置（零 API 消费）

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_gpt61_sol_dev.ps1 -Mode plan
python -m unittest discover -s gpt61_adapter_tests -p 'test_gpt61_adapter.py' -v
```

## 2. 先验证一次 API（不生成 GEANT 场景）

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_gpt61_sol_dev.ps1 -Mode probe
```

成功应显示 `PROBE RESULT: VALID_JSON_ACTION` 和真实的 `input_tokens`、`output_tokens`、`reasoning_tokens`、`cost_usd_estimate`。只调用 **1 次**模型；结果储存在 `gpt61_runs\connection_probe\usage.jsonl`。如果遇到 HTTP 400/404/403，先核对服务商的**真实 API 模型标识**、请求路由及密钥权限。

如服务商只实现了 Responses API：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_gpt61_sol_dev.ps1 -Mode probe -UpstreamApi responses
```

**注意：短 Prompt 的 probe 成本不足以预测真实 ICL Prompt 的价格。必须跑真实场景后再外推。**

## 3. 生成 20 个全新的开发验证场景（一次性）

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_gpt61_sol_dev.ps1 -Mode prepare
```

实际上调用的是原项目**已经用于 P1-C/P2-C 的场景生成器**：

```powershell
python p1b_generate.py --name gpt61_dev_seed7601 --split validation --seed 7601 --count 20
```

它可能较耗时，但只生成一次，后续重复执行会按 JSON manifest 和 `.pkl` 的 SHA-256 验证后直接复用。**不要将开发数据混入 Standard-1000 经验库，也不要使用 P2-C 的最终测试数据来选择模型。** 如果发生 `Ctrl+C` 中断且仅残留 `.pkl`，程序会拒绝覆盖；需先人工检查并备份。

## 4. 先运行 10 个场景、核算费用，再补足 20 个

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\run_gpt61_sol_dev.ps1 -Mode run -Scenes 10 -BudgetUsd 8
```

上面的 10 个场景会分成两段，每段 5 个场景，每个场景四个角色独立调用一次 GPT，共预期 40 次真实模型调用；对端点按实际 Token 用量计费。检查结果和估算：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_gpt61_sol_dev.ps1 -Mode costs
```

如果质量、耗时和费用都可接受，再**使用完全相同的模型参数**补足 20 个场景：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\run_gpt61_sol_dev.ps1 -Mode run -Scenes 20 -BudgetUsd 8
```

已验证成功的 0–9 场景自动跳过；新结果从第 10 个场景开始。已完整完成的 chunk 才会跳过，不完整/失败目录保留并拒绝覆盖。`-BudgetUsd 8` 是按**配置 Token 单价计算的预检消费保护线**，不是银行/服务商的真实账单硬限额；额外网关收费、失败请求无 usage、缓存写入费用和其他服务附加费可能不受这个限额约束。

如果前面用 `-UpstreamApi responses` 成功，正式 run **也必须写 `-UpstreamApi responses`**。如果默认模型别名不能识别，可用 `-UpstreamModel '服务商实际模型 ID'`，并在 probe 与 run 使用同一 ID。`-Effort medium` 默认，第一批不要调高；`-MaxOutputTokens 8192` 限制包括隐藏推理 Token 在内的输出额度。若全部用尽却没有产生可解析动作，运行器会停止，保留失败记录。不要临时放宽格式校验或让失败角色回退。

## 5. 输出文件

```text
gpt61_runs/
  connection_probe/usage.jsonl
  dev7601_iraware_std1000_v1/
    protocol_lock.json                   # 模型、数据、经验库、代码哈希及参数
    usage.jsonl                          # 每次调用 Token、隐藏推理 Token、时延、估算费用
    chunks/0000_005/{manifest.json,summary.json,metrics.csv,records.jsonl}
    chunks/0005_005/{...}
    ...
    dev20_summary.json                   # 完成的 n=10 或 n=20 场景平均指标
```

`records.jsonl` 仍保存原 `p1c_runner.py` 的角色决策与 Prompt；请勿把其中可能包含业务敏感信息的原始内容上传公共仓库。本包的 `usage.jsonl` 不记录密钥、Prompt 或完整响应。`costs` 会根据**真实返回的** usage 做 20、200、1536 场景费用外推。

## 6. 根据历史真实 Prompt 的事前费用测算

从你已完成的 P2-B 运行记录中提取，四角色合计平均约 **76,180 个提示词字符/场景**。仅作规划，按照 **4 字符约合 1 Token**，大约有 **19,045 输入 Token/场景**。GPT Tokenizer 和中文/JSON 标记会影响真实用量。默认官方标准单价为：每百万非缓存输入 $2、缓存输入 $0.10、输出 $10（输出包括隐藏 reasoning tokens；参考 OpenAI 官方 GPT-6.1 Sol 模型页面）。下面假设无缓存优惠，0.019045 百万输入 Token/场景；将输出用量分成情景，非预测值：

| 每角色平均输出 Token（含 reasoning） | 20 场景（80 次模型调用） | 200 场景（800 次调用） | 1536 场景（6144 次调用） |
|---:|---:|---:|---:|
| 500 | 约 $1.16 | 约 $11.62 | 约 $89.23 |
| 1500 | 约 $1.96 | 约 $19.62 | 约 $150.67 |
| 4000 | 约 $3.96 | 约 $39.62 | 约 $304.27 |

这些数字**不包括**真实服务商加价、缓存写入、特殊处理模式等收费。正式实验还有真实重试、失败消耗、输出令牌策略影响。**第一次 10 场景后，请以真实 usage 总量和服务商账单反推预算，不要基于上表盲目启动 200 场景。**

## 7. 重要限制与下一步

- 不知道当前网关是否将 `gpt-6.1-sol` 正确映射到官方模型。应以服务商模型查询与计费明细核验。
- 仅沿用你原来已运行成功的 **IR-aware + Standard-1000**，不混入风险重排序。相同数据的 Qwen 基线需要单独运行才能进行严格配对，且必须在新开发数据上进行。
- `p1c_runner.py` 的实际客户类源码没有随这个验证包一起提供。此代理按其公开暴露的 OpenAI Chat Completions 使用方式设计，离线测试覆盖请求转换与 JSON 回传，但还须通过你本机真实环境联调确认原客户端的全部行为。
- 无论 GPT 的 10–20 场景数值是否高于 MAPPO，均只代表 **技术与费用可行性预实验**，不能作为独立测试的统计结论。
- P2-C 的三个正式 seed 以及 12,288 条评分不能用于本轮模型选择。未来如果确定继续 GPT，应该新生成至少一套独立开发集选择参数，再用另一套完全保留的测试集确认性能。

> 安全性：`GPT61_API_KEY` 仅从当前环境变量读取；传输默认 HTTPS，不禁用 TLS 验证；本地代理只绑定 `127.0.0.1`。使用 `-AllowHttpUpstream` 会显著增加内网密钥暴露风险，应慎用。
