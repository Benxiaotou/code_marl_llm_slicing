# P1-E：IR-aware Qwen-ICL 的 2×2 消融实验

## 0. 目的与严格边界

这次**只新增两种运行方法**：

1. `ir_prompt_only`：**只保留 P1-D 完整 IR-aware 的风险指令**，不给出历史 IR 余量、历史成本和 g 历史效益等数字；不需要 `--ir-history`。
2. `ir_data_only`：**只保留 P1-D 的本角色历史 IR 数字以及必要的字段事实解释**，不增加 P1-D 的 IR 风险指令；必须提供原有 `--ir-history`。

配合已有 `icl` 和 `ir_icl`，形成标准四格消融：

| 方法 | 额外 IR 风险指令 | 历史本角色 IR 数据 | 运行结果 |
|---|:---:|:---:|---|
| `icl` | 否 | 否 | 原 P1-C 已完成，无需重跑 |
| `ir_prompt_only` | 是 | 否 | **本次新跑** |
| `ir_data_only` | 否 | 是 | **本次新跑** |
| `ir_icl` | 是 | 是 | 原 P1-D 已完成，无需重跑 |

P1-E 不修改冻结 Qwen 权重，不重新训练 MAPPO，不修改原 GEANT 环境、奖励或动作空间，不加 Guard。四个角色都只读取当前自己的 240D 局部观测，每个角色使用同一 P1-B 训练库内的相似三条案例。仅离线训练阶段的**本角色历史** IR 余量/成本可用于 `ir_data_only` 和已有 `ir_icl`。

**P1-D 的完整 `ir_icl` Prompt 保持原样。** 2×2 四种输入除两个因子以外，其余都保持一致：

- 普通 ICL **系统提示** == Data-only 系统提示。
- Prompt-only **系统提示** == 完整 IR-aware 系统提示。
- 普通 ICL **用户提示** == Prompt-only 用户提示。
- Data-only **用户提示** == 完整 IR-aware 用户提示。

所以不会发生换了历史案例、换了检索权重、换了 Prompt 基础内容等混杂因素。`ir_data_only` 的字段解释是原 P1-D 数据模块的一部分，保留其一致性；它没有新增任何“应保护 IR/降低让渡”的额外行为指令。

## 1. 安装和验证

前提：已完整安装此前 P1-A、P1-B、P1-C、P1-D，并保留：

- 历史训练库：`p1b_libraries/train_seed2601_v1/`（已审计，152 条/角色）
- P1-D 历史 IR overlay：`p1d_libraries/train_seed2601_ir_v1/`（已完成，不需要重建）
- 原验证场景：`p1b_data/p1c_val_seed4601.json`
- 原六方法结果目录：`p1c_runs/*` 和 `p1d_runs/val200_ir_icl3`

在 PowerShell：

```powershell
conda activate marl
cd F:\Projects\code_marl_v0_exper_1
```

先备份下列**将被替换**的 P1-D 文件，再解压本补丁到项目根目录：

```
p1c_runner.py
p1c_compare.py
p1c/policy.py
p1d/policy.py
```

补丁**新增** `p1e_ablation_report.py`、`tests/test_p1e.py`、`P1E_README.md`，不会覆盖 `main.py`、MAPPO checkpoint、历史经验库、验证数据或之前的结果目录。

离线测试：

```powershell
python -m unittest discover -s tests -p "test_p1e.py" -v
python -m unittest discover -s tests -p "test_*.py" -v
```

本补丁开发环境：新增 **10 项**测试通过；原 P1-A～P1-D 代码回归测试 **44 项**通过，总计 **54 项**。这是模拟环境和逻辑测试；真实 Qwen/GEANT 性能需要你在 Windows 环境执行。

## 2. 先做 5 场景接口试运行（推荐）

### A. Prompt-only（没有历史收益数据）

```powershell
python p1c_runner.py `
  --library .\p1b_libraries\train_seed2601_v1 `
  --dataset-manifest .\p1b_data\p1c_val_seed4601.json `
  --method ir_prompt_only --examples 3 `
  --start 0 --count 5 --seed 1 `
  --model qwen3-next-80b-a3b-instruct `
  --temperature 0.1 --max-tokens 4096 --api-timeout 120 --on-error abort `
  --output-dir .\p1e_runs\probe5_ir_prompt_only
```

注意：**此命令不能传 `--ir-history`**。该策略在运行时只读取原角色经验库，不读取 IR overlay，更不会读取 `private_offline`。

### B. Data-only（有历史收益数据，没有额外风险指令）

```powershell
python p1c_runner.py `
  --library .\p1b_libraries\train_seed2601_v1 `
  --ir-history .\p1d_libraries\train_seed2601_ir_v1 `
  --dataset-manifest .\p1b_data\p1c_val_seed4601.json `
  --method ir_data_only --examples 3 `
  --start 0 --count 5 --seed 1 `
  --model qwen3-next-80b-a3b-instruct `
  --temperature 0.1 --max-tokens 4096 --api-timeout 120 --on-error abort `
  --output-dir .\p1e_runs\probe5_ir_data_only
```

上面的两份试运行日志仅用于接口验证，**正式配对统计不能拿 5 个场景混进 200 场景**。

## 3. 在原来的同一批 200 场景完整运行

**A：IR-Prompt-only，200 个场景，共 800 次 Qwen 角色调用：**

```powershell
python p1c_runner.py `
  --library .\p1b_libraries\train_seed2601_v1 `
  --dataset-manifest .\p1b_data\p1c_val_seed4601.json `
  --method ir_prompt_only --examples 3 `
  --start 0 --count 200 --seed 1 `
  --model qwen3-next-80b-a3b-instruct `
  --temperature 0.1 --max-tokens 4096 --api-timeout 120 --on-error abort `
  --output-dir .\p1e_runs\val200_ir_prompt_only
```

**B：IR-Data-only，200 个场景，共 800 次 Qwen 角色调用：**

```powershell
python p1c_runner.py `
  --library .\p1b_libraries\train_seed2601_v1 `
  --ir-history .\p1d_libraries\train_seed2601_ir_v1 `
  --dataset-manifest .\p1b_data\p1c_val_seed4601.json `
  --method ir_data_only --examples 3 `
  --start 0 --count 200 --seed 1 `
  --model qwen3-next-80b-a3b-instruct `
  --temperature 0.1 --max-tokens 4096 --api-timeout 120 --on-error abort `
  --output-dir .\p1e_runs\val200_ir_data_only
```

两种方法会分别生成 `manifest.json`、`records.jsonl`、`metrics.csv`、`summary.json`。所有输出目录必须事先不存在或为空，程序拒绝覆盖历史实验记录。两组实验一共 **1600 次 Qwen 调用**。

请确保 `LLM_API_KEY`、`LLM_BASE_URL` 仍按 P1-C 方式设置，且模型、温度、stream、max tokens 与原 P1-D 实验相同。

## 4. 与原六种方法做八方法配对比较

```powershell
python p1c_compare.py `
  --knn .\p1c_runs\val200_knn `
  --icl .\p1c_runs\val200_icl3 `
  --ir-prompt-only .\p1e_runs\val200_ir_prompt_only `
  --ir-data-only .\p1e_runs\val200_ir_data_only `
  --ir-icl .\p1d_runs\val200_ir_icl3 `
  --mappo .\p1c_runs\val200_mappo `
  --random-icl .\p1c_runs\val200_random_icl3 `
  --zero .\p1c_runs\val200_zero `
  --bootstrap 5000 `
  --output-dir .\p1e_runs\compare_val200_all8
```

完整结果在：

```
p1e_runs/compare_val200_all8/
  comparison.json
  paired_scenes.csv
```

比较器拒绝以下情况：训练场景混入验证集、数据集/环境哈希不匹配、不同 Qwen 参数、Mock/Guard/回退记录、任一角色检索历史案例 ID/距离不同、`ir_data_only` 与 `ir_icl` 的历史反馈来源哈希不同、或者四格输入未做到严格的提示一致性。

**四格比较（重点查看）：**

- `ir_prompt_only_vs_icl`：仅额外 IR 风险指令的影响；
- `ir_data_only_vs_icl`：仅增加本角色历史 IR 信息的影响；
- `ir_icl_vs_ir_prompt_only`：已知风险指令时，再增加历史 IR 数据的影响；
- `ir_icl_vs_ir_data_only`：已知历史 IR 数据时，再增加风险指令的影响；
- `ir_ablation_factorial.per_metric[*].factor_interaction`：**完整 IR - Data-only - Prompt-only + 普通 ICL**，用于检查两个因子叠加是否存在额外相互作用（每场景独立计算后配对 Bootstrap）。

生成精简 Markdown 报告：

```powershell
python p1e_ablation_report.py `
  --comparison .\p1e_runs\compare_val200_all8\comparison.json `
  --output .\p1e_runs\compare_val200_all8\P1E_ABLATION_REPORT.md
```

## 5. 建议如何解释结果

核心指标优先级：**合作有效率、个体理性违约、系统效益 V、成本，再看接纳率和延迟**。

你已经得到的同场景开发基线：普通 ICL 接纳率 93%、合作有效率 66%、V 336.834、成本 44.285；完整 IR-aware 接纳率 95%、合作有效率 74.5%、V 342.407、成本 38.635；MAPPO 合作有效率 82.5%、V 346.053。

请等待新两种消融的**真实运行结果**后，再对改善来源做判断。只看 Prompt-only 和 Data-only 的某一个指标还不够，要结合四格交互项与逐场景差异，且不能将历史 IR 余量解释为单个动作的因果收益。

此处仍属于开发验证集：之前已经多次观察这 200 个场景，不能再声称它们是未使用的最终测试集。最终论文仍应冻结方案、在全新独立测试集上重新评估，最好重复 Qwen 采样以估计模型随机性。
