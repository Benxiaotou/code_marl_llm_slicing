# P1-D：IR-aware Qwen-ICL（历史个体理性感知）

## 实验目的

在**相同的 `p1c_val_seed4601` 200 个场景**上，比较新增 `IR-aware Qwen-ICL` 与原先五种方法：冻结 MAPPO、kNN-Action、Qwen-Coop-ICL、Random-ICL、Zero-shot。

**这次只需要新运行 IR-aware 一种方法。** 原 P1-C 200 场景五种完整结果不需要重跑，不重新训练 MAPPO、不修改主环境、不调整奖励、不使用 Guard、不在测试时试运行搜索动作，也不微调 Qwen。

> 研究问题：固定局部观测、数值距离、3 条相似历史示例、模型和采样配置后，**额外提供本角色历史合作收益余量/调整成本**能否提高合作有效率，缩小与 MAPPO 的差距？

这是 **validation/dev 实验**。这些 200 个场景已经用于 P1-C 比较与方向选择，不应作为最终未见测试集。论文最终结论仍需新的独立测试集和必要的重复采样。

## 1. 安装补丁

在 Windows PowerShell：

```powershell
conda activate marl
cd F:\Projects\code_marl_v0_exper_1
```

将压缩包解压到项目根目录：新增 `p1d/`、`p1d_build_ir.py`、`p1d_report.py`、`tests/test_p1d.py` 和此说明，**更新** `p1c/policy.py`、`p1c_runner.py`、`p1c_compare.py`。请先保留这三个原文件备份；补丁不覆盖 `main.py`、checkpoint、P1-B 经验库或原实验记录。

测试：

```powershell
python -m unittest discover -s tests -p "test_p1d.py" -v
python -m unittest discover -s tests -p "test_*.py" -v
```

本次本地开发环境里 **9 个新增测试、53 个既有测试，共 62 个通过**。实际 MAPPO checkpoint 与 Qwen API 测试需在你的本地 Windows 项目中执行。

## 2. 离线构建 IR 历史信息（只需运行一次）

```powershell
python p1d_build_ir.py `
  --library .\p1b_libraries\train_seed2601_v1 `
  --output-dir .\p1d_libraries\train_seed2601_ir_v1
```

从 P1-B 训练库的 `private_offline/selected_joint_records.jsonl` 读取**历史训练**联合结果，与 152 个已审计经验 ID、240 维观测、教师动作逐条核对。输出单独目录，不改变原 P1-B 库：

```
p1d_libraries/train_seed2601_ir_v1/
  ir_manifest.json
  ir_audit.json
  role_ir_history/
    agent_k1.jsonl
    agent_k2.jsonl
    agent_k3.jsonl
    agent_g.jsonl
```

### IR 新字段定义

- 对 k1/k2/k3：历史自身 IR 余量 `V_ki - V_ki_0`、余量相对于合作前效益的比例、该切片历史调整成本 `cost_ki`。
- 对 g：仅保存 g 在**历史联合决策**中的自身效益 `V_g`。原环境没有相应的 `cost_g` 或 `V_g_0`，因此不虚构。

**绝不向运行时 Prompt 提供：**历史其他切片收益/成本、当前其他切片状态、当前系统效益、全局物理网络对象、历史联合动作或所有切片的奖励汇总。系统只利用训练历史中的本角色数字提示行动风险。

**重要解释：**历史 IR 余量是当时四个智能体联合动作后的结果；它不是本角色动作的因果效果，也不能直接预测当前场景的真实 IR 值。

如果没有 `private_offline/selected_joint_records.jsonl`，请检查 P1-B 的选定经验库是否完整；不能用当前验证场景的效益或五组历史验证结果替代，以免数据泄漏。

## 3. 单场景 API 联调（可选，推荐）

```powershell
python p1c_runner.py `
  --library .\p1b_libraries\train_seed2601_v1 `
  --ir-history .\p1d_libraries\train_seed2601_ir_v1 `
  --dataset-manifest .\p1b_data\p1c_val_seed4601.json `
  --method ir_icl --examples 3 `
  --start 0 --count 5 --seed 1 `
  --model qwen3-next-80b-a3b-instruct `
  --temperature 0.1 --max-tokens 4096 --api-timeout 120 `
  --on-error abort `
  --output-dir .\p1d_runs\probe5_ir_icl
```

使用此前相同 `LLM_API_KEY` 和 `LLM_BASE_URL` 环境变量。注意，此 probe 与完整 200 场景测试使用**不同输出目录**。

## 4. 正式运行 200 场景 IR-aware Qwen-ICL

```powershell
python p1c_runner.py `
  --library .\p1b_libraries\train_seed2601_v1 `
  --ir-history .\p1d_libraries\train_seed2601_ir_v1 `
  --dataset-manifest .\p1b_data\p1c_val_seed4601.json `
  --method ir_icl --examples 3 `
  --start 0 --count 200 --seed 1 `
  --model qwen3-next-80b-a3b-instruct `
  --temperature 0.1 --max-tokens 4096 --api-timeout 120 `
  --on-error abort `
  --output-dir .\p1d_runs\val200_ir_icl3
```

这将调用 Qwen 200×4=**800 次**，并输出与原 P1-C 相同字段格式的 `manifest.json`、`records.jsonl`、`summary.json`、`metrics.csv`。程序拒绝覆盖现有输出目录。

**信息边界：**每个智能体只使用自己的**当前局部观测**、同一角色在冻结 MAPPO 训练库中检索的 3 条原始示例、这些历史案例对应的**本角色历史 IR 反馈**。检索算法完全与 `icl` 相同，动作空间仍为 4+4+4+3，环境直接执行生成动作。

## 5. 与 P1-C 现有五种方法做配对比较

保留原来完整的五个结果目录：

- `p1c_runs/val200_mappo`
- `p1c_runs/val200_knn`
- `p1c_runs/val200_icl3`
- `p1c_runs/val200_random_icl3`
- `p1c_runs/val200_zero`

运行：

```powershell
python p1c_compare.py `
  --knn .\p1c_runs\val200_knn `
  --icl .\p1c_runs\val200_icl3 `
  --ir-icl .\p1d_runs\val200_ir_icl3 `
  --mappo .\p1c_runs\val200_mappo `
  --random-icl .\p1c_runs\val200_random_icl3 `
  --zero .\p1c_runs\val200_zero `
  --bootstrap 5000 `
  --output-dir .\p1d_runs\compare_val200_all6
```

比较程序会校验数据集 ID/哈希、场景 ID、场景种子、原环境代码和配置哈希、训练经验库哈希、模型/采样参数，以及**IR-aware 与 ICL 的每个角色检索案例 ID 和距离完全一致**。如不一致，则拒绝生成对比结果，而不是给出不公平结果。

结果文件：`comparison.json`、`paired_scenes.csv`。

生成精简可读的实验报告：

```powershell
python p1d_report.py `
  --comparison .\p1d_runs\compare_val200_all6\comparison.json `
  --output .\p1d_runs\compare_val200_all6\P1D_RESULT_SUMMARY.md
```

## 6. 重点看哪些指标？

以历史 P1-C 200 场景基线为参照：

| 方法 | 接纳率 | 合作有效率 | 平均 V | 平均调整成本 |
|---|---:|---:|---:|---:|
| MAPPO | 85.5% | 82.5% | 346.053 | 34.591 |
| Qwen-Coop-ICL | 93.0% | 66.0% | 336.834 | 44.285 |
| kNN-Action | 85.5% | 62.5% | 332.075 | 44.769 |
| Random-ICL | 92.0% | 20.5% | 282.353 | 85.241 |
| Zero-shot | 22.0% | 0% | 269.010 | 32.143 |
| **IR-aware Qwen-ICL** | **待测** | **待测** | **待测** | **待测** |

优先观察：**合作有效率、已有切片 IR 违约场景数量、V、成本**，再看接纳率和延迟。已有 P1-C 的合作有效率 66% 距离 MAPPO 82.5% 还差 16.5 个百分点，因此 IR-aware 的目的是缩小这一差距，而非只提高接纳率。

若 IR-aware 显著改善合作有效率且不明显降低接纳率，有希望支持“角色历史收益信息对冻结 LLM 决策有帮助”。但如果复杂度上升且接近/不如 kNN 等数值方法，则需要客观报告方法价值与限制。

### 论文边界

1. 这批 200 场景已被多次用于 P1-C 开发，因此此次实验仍是**同场景验证/消融**，不是全新独立测试。
2. 5 组原结果每场景只有一次模型采样；对随机输出模型，正式论文建议至少做多次独立采样和全新的最终测试集。
3. 没有真实内网 Qwen 接入能力的本环境只完成离线代码与结构测试，**不提供虚假的新接纳率或效益**。
4. 如后续改变 `main.py` 或修复 g 压缩/准入逻辑，必须全方法重新统一评估，不能直接沿用旧日志。
