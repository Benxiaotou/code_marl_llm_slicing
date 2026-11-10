# P2-D v1.2：节点 vs 链路本地反事实消融

## 目的
使用 GPT-6.1 Sol 与冻结 MAPPO 在 `gpt61_dev_seed7601` 的 scene 0～9 中已经保存的环境动作，对三个已有切片 `k1/k2/k3` 的节点控制、链路控制进行分组替换，复用原 GEANT 环境重新评分。不发起新模型/API 请求。

## 六组动作
| 组别 | 已有切片 k1/k2/k3 | 新切片 g |
|---|---|---|
| MM | MAPPO 的完整四维动作 | MAPPO |
| GG | GPT 的完整四维动作 | GPT |
| MG | MAPPO 的完整四维动作 | GPT |
| GM | GPT 的完整四维动作 | MAPPO |
| NODE_M | **节点维 0/1 来自 MAPPO，链路维 2/3 来自 GPT** | GPT |
| LINK_M | **节点维 0/1 来自 GPT，链路维 2/3 来自 MAPPO** | GPT |

所有新分组均逐维使用已缓存数值，不更改第四维、不增加新动作或模型推理。指标包括接纳率、合作有效率、系统效益 V、新切片 V_g、成本和各已有切片 IR 余量。

## 环境
- Windows PowerShell、Conda `marl`
- 项目根目录：`path/to/code_marl_llm_slicing`
- 必须已存在的原始诊断包：`GPT61_MAPPO_10scene_diagnosis.zip`
- 场景清单：`p1b_data\gpt61_dev_seed7601.json`
- 历史库：`p2b_libraries\p2b_standard_1000_v2`
- 原项目的 `main.py`、`p1a_runner.py`、`p1c_runner.py`、冻结 MAPPO checkpoint 保持不变

## 使用（PowerShell）
先将下载的 `P2D_Node_Link_Ablation_v1_2.zip` 放到项目根目录，执行：

```powershell
conda activate marl
cd path/to/code_marl_llm_slicing

# 解压到新文件夹：不覆盖已成功的 P2-D v1.1
Expand-Archive .\P2D_Node_Link_Ablation_v1_2.zip `
  -DestinationPath .\p2d_node_link `
  -Force

# 只检查记录结构和动作；不启动 GEANT
python .\p2d_node_link\p2d_counterfactual_replay.py `
  --preflight `
  --archive .\GPT61_MAPPO_10scene_diagnosis.zip

# 运行六组，每组 10 个同场景；不调 GPT/Qwen API
python .\p2d_node_link\p2d_counterfactual_replay.py `
  --project-root . `
  --archive .\GPT61_MAPPO_10scene_diagnosis.zip `
  --dataset-manifest p1b_data\gpt61_dev_seed7601.json `
  --library p2b_libraries\p2b_standard_1000_v2 `
  --seed 1 `
  --output-dir gpt61_runs\dev7601_node_link_ablation_v1_2

# 查看六组汇总、哪些场景因替换节点/链路而恢复接纳
Get-Content .\gpt61_runs\dev7601_node_link_ablation_v1_2\summary.json -Raw
```

**不需要提供 API Key，本程序不会调用 GPT 或 Qwen API。**

## 输出文件
- `summary.json`：六组接纳、合作有效率、V、V_g、成本；节点/链路恢复接纳的场景编号与分类。
- `paired_crossover_by_scene.csv`：10 场景 × 六组的配对结果。
- `scene_metrics.csv`：六组每场景的 IR 余量等指标。
- `{CASE}_records.jsonl`：评分后的详细记录与动作来源。
- `protocol.json`：样本 SHA-256、代码/配置/checkpoint 哈希、动作维度来源等，可用于论文复现。
- 若对照不一致，写 `CONTROL_FAILED.json` 并立即中止，不产生有效 `summary.json`。

## 校验与解释
程序会首先重放 MM、GG，逐场景核对原始指标；再核对 MG、GM 的平均结果是否与已验证的 v1.1 一致。只有通过后才进行 NODE_M、LINK_M 消融，并标记 `status=VALIDATED`。

已知的 v1.1 结果：MM 接纳 6/10、GG 接纳 0/10；MG 接纳 6/10（V≈318.669）、GM 接纳 0/10（V≈266.266）。

判断节点/链路的贡献时，重点看原 GG 失败、MG 成功的场景：
- `NODE_M` 成功而 `LINK_M` 失败：单独替换节点控制足以恢复该场景接纳；
- `LINK_M` 成功而 `NODE_M` 失败：单独替换链路控制足以恢复；
- 两者都成功：任一单组替换均可恢复；
- 两者都失败但 MG 成功：需要节点与链路共同替换（存在交互依赖）。

**谨慎性**：上述结论仅描述这 10 个开发场景的动作干预；由于节点与链路可能相互作用，不应将其推广为 GPT 失败的普遍单一原因，也不能用该批数据进行独立统计验证。

## 兼容性说明
v1.2 保留了 v1.1 的 `__main__.InitVnetworkTrain` 旧 pickle 兼容注册步骤，避免再次出现 `Can't get attribute 'InitVnetworkTrain'`。
旧版 v1.1 代码与已产生的验证结果不会被覆盖。本包可选运行离线单元测试：

```powershell
python -m unittest discover -s .\p2d_node_link -p test_p2d_counterfactual_replay.py -v
```

如果重放中途出现异常，请把完整 Traceback 和 `CONTROL_FAILED.json` 一并发回，不要覆盖已有实验结果。
