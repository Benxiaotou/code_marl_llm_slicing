# P2-B 新开发集配对分析与串行批测工具包

## 已完成的首两组对照

输入：2026-11-01 上传的两份 `val5601_base152_p2a200.zip` 和 `val5601_standard300_p2a200.zip`。

- 均使用 `p2b_val_seed5601`、相同 SHA256 `7d92e97808f3269dcf11b3275c1cf85f6daae50d1678f412b7c29e347ad79c86`、200 场景、相同冻结 Qwen 和 P2-A 参数。
- `Base-152`: acceptance 91.5%，合作有效率 74.5%，V=341.2544，cost=38.2330。
- `Standard-300`: acceptance 88.0%，合作有效率 77.0%，V=341.5403，cost=35.6455。
- 同场景配对：新增合作有效 20、退化 15；合作有效率 +2.5 pp、Bootstrap 95% CI 约 [−3.0,+8.5] pp；McNemar exact p≈0.500。
- 平均成本下降 2.588，Bootstrap 95% CI 约 [−3.99,−1.07]，配对 t 检验 p≈0.00067；V 的均值差 +0.286，Bootstrap CI [−3.56,+4.05]。
- 解释：合作有效率暂不能断定稳健提升，成本下降值得后续验证。此处数据仅为**开发验证集**，不是最终独立测试结论。
- 附带 `paired_base152_vs_standard300.csv` 与 `paired_summary.json`，方便后续绘图与论文整理。

## 一次串行跑完当前首轮 18 组实验

只需从本包取出两个文件：

- `run_p2b_val5601.ps1`
- `p2b_run_val5601_queue.py`

复制到 `path/to/code_marl_llm_slicing` 根目录，保持现有经验库、IR 历史、数据集、`p1c_runner.py` 不变。

在 **PowerShell 已激活 marl** 环境中先执行只读预检（不会启动实验）：

```powershell
cd path/to/code_marl_llm_slicing
conda activate marl
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_p2b_val5601.ps1
```

确认最后看到 `PRECHECK OK`，再使用**唯一的一条命令**串行跑完尚未完成的任务：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_p2b_val5601.ps1 -Execute
```

### 自动安排的 18 组

| 方法 | 经验库 | 次数 |
|---|---|---:|
| kNN（不调用 Qwen） | Base-152、Standard-300/500/1000、Balanced-300/500/1000 | 7 |
| 冻结 MAPPO | 新 val5601 数据集一次（使用 Base 库满足 runner 输入合同） | 1 |
| P2-A `ir_rerank` | 上述七库（Base-152、Standard-300 已完成后自动校验跳过） | 7 |
| P1-D `ir_icl` | Base-152、Standard-500、Balanced-500 | 3 |

如果此前两组完整结果仍在 `p2b_runs/val5601_base152_p2a200` 和 `p2b_runs/val5601_standard300_p2a200`，并且所有记录与哈希都能通过校验，首轮**预计还剩 16 个任务**。若它们被移动、修改或不完整，预检会明确提示，**绝不擅自重跑或覆盖**。

可选：将 P1-D 追加到全部七库（总计 22 组）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_p2b_val5601.ps1 -ExtendedIR -Execute
```

也可以先只跑 Qwen P2-A：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_p2b_val5601.ps1 -Scope rerank -Execute
```

### 安全约束

1. **必须**在已验证的项目原目录运行，使用已安装的 `marl` 环境；脚本调用当前 `python`，不会安装软件或连接任何额外账户。
2. 先核查验证数据集原始样本 SHA、原主程序和配置 SHA、经验库与 IR manifest SHA、每个角色经验文件 SHA，再检查已有结果是否完整。
3. **严格复用**原实验的 Qwen 模型、`seed=1`、`temperature=0.1`、`examples=3`、`max_tokens=4096` 和 P2-A 风险参数；使用库专属的 IR 历史，kNN 和 MAPPO 不使用 IR。
4. 每个任务调用 `p1c_runner.py` 的 `--on-error abort`，程序退出非零即停止；若产出失败、fallback、只完成部分 200 场景或出现非预期模型/数据 SHA，也停止后续队列。
5. 只有当前输出目录的 `manifest.json`、`summary.json`、`metrics.csv` **以及完整 `records.jsonl`** 均存在，且 200 场景校验、哈希及指标吻合，才能显示 `SKIP VERIFIED`。你先前上传的两份精简 ZIP **不含** `records.jsonl`，但原 `p1c_runner.py` 正常输出目录应包含它；请不要用精简 ZIP 替换本地原始结果目录。
6. 已存在**不完整**的运行目录会阻止继续，不会被自动修复、删除、覆盖。重新执行脚本会跳过已经验证完成的任务；需人工检查失败目录，并在备份后按项目规范处理。
7. 日志保存在 `p2b_runs/p2b_val5601_queue_logs/`，每组独立 `.log`，任务事件追加在 `queue_events.jsonl` 中。运行结束后输出 `SUCCESS`。
8. 远程大模型服务的可用性、速率限制、机器不休眠、断电/网络中断均无法由本脚本保证。若需全程无人值守，请先运行预检，确保供电和网络稳定。
9. 每个任务使用固定 200 场景，所有模型表现统计仅用 **新开发验证集**。保留最终独立测试集，不在此脚本中执行。

### 执行完后交付什么

上传各新任务的 `manifest.json`、`summary.json`、`metrics.csv`；有条件可以附带 `records.jsonl` 或统一 ZIP 归档。请保留整个 `p2b_runs/` 本地原始日志，以便复现。

## 验证范围

此工具基于你真实上传的 `p1c_runner.py` 源码和两个真实 200 场景输出设计；已对跳过校验、参数一致性、坏数据拒绝及任务数量做了 9 项本地单元测试。
**PowerShell 入口及真正 16 组剩余任务没有在本机 Windows + Qwen + GEANT 环境执行**，必须先让它在你的本地项目中完成只读预检。
