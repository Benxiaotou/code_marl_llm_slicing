# P2-C 冻结 Qwen：八方法 × 三测试种子串行实验工具

## 1. 本次正式测试协议

只使用 `qwen3-next-80b-a3b-instruct`（冻结，不微调）。

- **正式测试种子**：`6601`、`6701`、`6801`；每个新生成 **512** 个场景，总计 **1,536** 个。
- **每个种子只生成一次场景**。所有八种方法复用该种子所对应的完全相同的场景文件。
- **分段大小**：每段 **32** 个场景（16 段/方法/种子）。
- **总批次**：8 方法 × 3 种子 × 16 段 = **384** 个严格串行任务。
- **场景级评价**：8 × 1,536 = **12,288** 次。
- **Qwen 评价**：六种方法 × 1,536 = **9,216** 次场景决策，即四角色串行调用 **36,864** 次。

### 固定的八种方法

| 名称（结果目录） | 原 runner 方法 | 经验库 | 目的 |
|---|---|---|---|
| `mappo` | `mappo` | Base-152 占位参数（实际使用冻结教师 checkpoint） | 教师基线 |
| `knn_std1000` | `knn` | Standard-1000 | 纯经验动作复用 |
| `icl_std1000` | `icl` | Standard-1000 | **本次新增加的普通 ICL** |
| `iraware_std1000` | `ir_icl` | Standard-1000 + 对应 IR overlay | IR 反馈贡献 |
| `p2a_std1000` | `ir_rerank` | Standard-1000 + IR overlay | 主要候选 |
| `p2a_base152` | `ir_rerank` | Base-152 + 对应 IR overlay | 小经验库基线 |
| `p2a_std500` | `ir_rerank` | Standard-500 + 对应 IR overlay | 成本最佳候选 |
| `p2a_bal1000` | `ir_rerank` | Balanced-1000 + 对应 IR overlay | 同规模经验质量对照 |

冻结的推理设置：`--seed 1`、`--temperature 0.1`、`--max-tokens 4096`、`--examples 3`、`--on-error abort`；P2-A 风险配置候选池 20、权重 0.35、距离差 0.10、距离比 1.35；无 Guard、无 Fallback、无在线跨智能体通信。

## 2. 文件安装

ZIP 文件包含：

- `p2c_qwen8_queue.py`：场景复用、队列、校验、续跑与配对统计的主程序。
- `run_p2c_qwen8.ps1`：Windows PowerShell 入口。
- `README_P2C_Qwen8_中文.md`：本文。
- `tests/test_p2c_qwen8.py`：离线模拟环境回归测试，不会联网或调用 Qwen。

将上述文件解压到**原始实验项目根目录**：

`path/to/code_marl_llm_slicing`

不要放入发布仓库 `code_marl_llm_slicing_publish` 执行。项目中既有的 `p1b_generate.py`、`p1c_runner.py`、`main.py`、`configs_main.yaml`、`p1a_runner.py`、经验库、IR overlays 和 MAPPO 模型文件应继续保留。

> 这份队列基于此前已上传并成功用于 P2-B 的 `p1c_runner.py` 和 P2-B 队列设计；**没有拿到你当前 Windows 本地 `p1b_generate.py` 的完整源码**。请务必先做以下只读预检，确认生成器接受 `--split test`。如果报参数错误，不要把测试数据改成 `train` 或 `validation`，请保留输出并反馈。

## 3. 推荐：一个命令串行完成场景生成 + 八方法实验

上一版把生成和评价拆成 `prepare` 与 `execute`，对长时间无人值守任务不方便。
**新版增加 `-Mode all`，在同一进程中依次执行：**

1. 检查 6601 数据集：已有完整有效数据则复用；否则生成 512 个场景并校验哈希。
2. 对 6701、6801 做同样处理。所有场景生成任务严格串行，绝不重复生成已验收的数据。
3. 自动锁定三套数据集及代码/经验库哈希；启动 384 段串行评价。
4. 所有分段通过验收后自动生成最终统计 CSV。

```powershell
conda activate marl
cd path/to/code_marl_llm_slicing

powershell -NoProfile -ExecutionPolicy Bypass -File .\run_p2c_qwen8.ps1 -Mode all
```

旧的 `-Mode plan`、`prepare`、`execute`、`aggregate` 保留，便于只检查或单独执行某个步骤。
**现在不需要手动先运行 prepare 再执行 execute。**

### 对 Ctrl+C 中断生成的特别提醒

- `p1b_generate.py` 是原来的场景生成器。队列能恢复**已完整保存并校验通过的数据集**和**成功完成的 32 场景评价段**；
  但如果原生成器没有中间检查点，那么正生成中的 512 场景在被中断后**不能保证从中断处继续**。
- 日志显示你在 6601 的 XML 流量文件读取过程中按了 Ctrl+C。先检查
  `p1b_data\p2c_test_seed6601.json` 和 `data\GEANT\p2c_test_seed6601.pkl` 是否存在。
- 如果 `.json` 存在，`-Mode all` 会先进行完整校验，通过则直接复用。
- 如果只有 `.pkl` 而没有有效 `.json`，不能把该 `.pkl` 直接当作 512 个完整场景。
  新版会**主动停止并提示检查**，绝不直接覆盖它。请先将残留文件归档备份，再重试。
- 新增生成日志：`p2c_runs\qwen8_test_v1\logs\generation\p2c_test_seedXXXX.log`，
  包括终端未打印的 GEANT 详细信息。可使用 `Get-Content <日志路径> -Tail 30 -Wait` 查看。
- 如果你想进一步实现**同一 512 场景生成过程的逐场景检查点/续跑**，
  需要检查并改造你本地的 `p1b_generate.py`，而不仅是改队列入口。
  未读取生成器源码前，不应直接拼接部分随机场景，因为这可能破坏既定数据集生成协议。

## 4. 旧版分阶段 PowerShell 命令（可选）

打开已激活 `marl` 的 PowerShell：

```powershell
conda activate marl
cd path/to/code_marl_llm_slicing

# 可选：确认 p1b_generate.py 支持 test 类型（只显示帮助）
python .\p1b_generate.py --help

# 第一步：只读预检查；不生成新场景、不触发 Qwen
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_p2c_qwen8.ps1 -Mode plan

# 第二步：生成 3 个新测试集并完成 SHA256 校验；已有且校验正确的将复用
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_p2c_qwen8.ps1 -Mode prepare

# 第三步：开始 384 个严格串行测试任务；成功完成的分段会自动验证并跳过
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_p2c_qwen8.ps1 -Mode execute

# 可选：完成后独立重新汇总
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_p2c_qwen8.ps1 -Mode aggregate
```

不想使用 PowerShell 包装脚本，也可以直接执行：

```powershell
python .\p2c_qwen8_queue.py --mode plan
python .\p2c_qwen8_queue.py --mode prepare
python .\p2c_qwen8_queue.py --mode execute
python .\p2c_qwen8_queue.py --mode aggregate
```

### 正式测试前的一次轻量联调（只用开发集）

由于 Standard-1000 普通 ICL 是本次新增的对照，建议先在**现有开发集 val5601** 上确认接口能运行，不在正式测试集上调参数：

```powershell
python .\p1c_runner.py `
  --library .\p2b_libraries\p2b_standard_1000_v2 `
  --dataset-manifest .\p1b_data\p2b_val_seed5601.json `
  --method icl --examples 3 `
  --start 0 --count 1 --seed 1 `
  --model qwen3-next-80b-a3b-instruct `
  --temperature 0.1 --max-tokens 4096 `
  --api-timeout 120 --on-error abort `
  --output-dir .\p2b_runs\val5601_std1000_icl_smoke1_p2c
```

该烟测输出目录必须不存在；如果之前已运行过该目录，请改用另一个全新目录。确认 `n_scored=1`、`n_failed_policy=0` 后，开始正式测试。

## 5. 断线续跑与失败保护

- 一次仅执行一个 `p1c_runner.py` 进程，不并发调用内网 Qwen。
- 每段单独记录 `manifest.json`、`summary.json`、`metrics.csv`、`records.jsonl`。
- 重新执行 `--mode execute` 时，只有**原始记录、场景 ID、哈希、配置、评分数均一致且没有失败**的整个段才会被 `SKIP VERIFIED` 跳过。
- 网络错误、程序退出码异常、模型输出解析失败、策略失败、缺少数据或摘要与原始记录不一致时，立即停止并保留所有原始证据。**不会自动跳过失败场景或重试挑选更好的决策结果。**
- 如果某段留下了不完整的输出目录，需要**先审计失败原因**，把整个失败目录**移动到归档备份目录**，保留失败日志，之后才能重启队列从该段重新执行。不要覆盖或直接删除原始结果。
- 生成过程中留下不完整的样本文件，同样不会被自动覆盖或重新生成；先检查文件状态。
- 全部大规模任务期间不要修改 Python 源码、经验库、IR 记录、模型 checkpoint、样本文件或环境配置；队列会锁定其哈希，并在重新启动时核验。

示例：假设失败的确实是 `6601 / icl_std1000 / chunk_0032_032`，在确认故障并保存诊断信息后执行以下归档操作：

```powershell
$root = "path/to/code_marl_llm_slicing"
$failed = Join-Path $root "p2c_runs\qwen8_test_v1\p2c_test_seed6601\icl_std1000\chunk_0032_032"
$archive = Join-Path $root "p2c_failed_attempts\p2c_test_seed6601_icl_std1000_chunk_0032_032_$(Get-Date -Format yyyyMMdd_HHmmss)"
Move-Item -LiteralPath $failed -Destination $archive
# 然后重新执行 --mode execute；已经验证完成的其他分段会自动跳过。
```

运行机器应保持供电、网络与 Qwen 服务连接稳定，避免 Windows 休眠；队列并不保证机器断电时能继续执行，但可从已经成功的段后恢复。

## 6. 输出文件

输出根目录：`p2c_runs/qwen8_test_v1/`

| 文件 | 作用 |
|---|---|
| `protocol_lock.json` | 冻结的任务协议、场景 SHA、关键代码/经验库/模型 SHA |
| `p2c_test_seed6601/<方法>/chunk_0000_032/` | 每段原始实验结果（其他种子、其他段以此类推） |
| `logs/<种子>/<方法>/<起始场景>.log` | 原始运行日志 |
| `queue_events.jsonl` | 每段开始、通过、失败的事件记录 |
| `p2c_all_scene_metrics.csv` | 所有 12,288 场景级指标，含场景配对键 |
| `p2c_summary_by_seed_and_pooled.csv` | 每个种子及三种子合并的八方法汇总 |
| `p2c_paired_comparisons.csv` | 九项预设配对对照、成功场景胜负翻转及百分位 Bootstrap 区间 |

只在 **384 段全部通过验收后**输出最终合并统计。

## 7. 论文中如何报告

- 全部三个种子均作为预先确定的正式测试集，不能挑最好看的种子报告。
- 评价、检索、输出与统计完整区分 `acceptance_rate` 与 `cooperative_valid_rate`。
- `p2c_paired_comparisons.csv` 的配对单位是同一个 `(dataset_seed, scene_id)`；总体 Bootstrap 在各预设种子内分层抽样。
- 该置信区间反映**场景采样**的不确定性，**不自动覆盖 Qwen 同场景重复调用的随机波动**；当前只有三个场景生成种子，跨环境分布推断应谨慎。
- 先前 `val4601` 和 `val5601` 仅保留为开发验证，不合并进入本次正式测试。
- 本实验测试的是**相同 GEANT 场景生成机制下**的独立样本泛化，不是跨拓扑泛化。

## 8. 本包已验证与尚未验证的范围

本包通过了 **Python 语法编译、离线模拟场景生成与八方法队列完整运行、重启跳过完整结果、拒绝不完整结果、拒绝不完整生成文件**等测试。

本地测试采用**模拟数据与模拟运行器**，没有调用真实 GEANT、MAPPO checkpoint 或内网 Qwen，因此**不能据此认为真实项目的实验已经完成**。

如 `--mode plan` 或 `--mode prepare` 在你本地报错，请保留报错信息；优先解决接口兼容性，不要擅自修改模型方案或把训练/开发集代替正式测试集。
