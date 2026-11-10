# P2-B 第二阶段：经验库规模与分层均衡实验

## 本阶段完成什么

本工具包会从本机已有的两个经过 P1-B 审计的经验库读取数据：

- 原始 152 组：`p1b_libraries/train_seed2601_v1`
- 新增 1,209 组：`p2b_libraries/p2b_train_seed2602_candidate_v1`

在不改变旧库的情况下，建立六个新版本：

| 系列 | 300 组 | 500 组 | 1,000 组 |
|---|---|---|---|
| `standard` | 原 152 + 新 148 | 原 152 + 新 348 | 原 152 + 新 848 |
| `balanced` | 原 152 + 新 148 | 原 152 + 新 348 | 原 152 + 新 848 |

旧 152 组固定为每个库的前缀，后续 300 → 500 → 1,000 在各自系列内部严格嵌套。

**standard**：根据新候选库已经存在的 `selection_quality_score` 从高到低选择，作为“优先选高评分经验”的基线。

**balanced**：使用原 P1-B 已存储的 `selection_stratum`，优先补齐旧库覆盖较少的分层；在分层内部优先使用高评分经验。不同规模之间仍然嵌套。

注意：此处的均衡是**原 P1-B 分层均衡**，不是新实现的 IR 余量/调整成本多维优化。后者仍需核对 `p1b/select.py` 和实际私有联合记录后再精细化设计。两系列的数量相同，所以可以开展初步筛选质量对照。

## 输入要求

必须保留原有文件结构，两个输入库中都需存在：

- `library_manifest.json`
- `audit_report.json`
- `private_offline/selected_joint_records.jsonl`
- `role_libraries/agent_k1.jsonl`、`agent_k2.jsonl`、`agent_k3.jsonl`、`agent_g.jsonl`

程序直接在你的 Windows 项目里读取这些文件，不需要上传整份私有联合经验给我。

它将检查来源、计数、四角色对应关系、局部状态和动作的格式、原教师追踪信息、两份库是否存在 ID 碰撞、相同的四角色观测，及旧验证集的来源 ID/SHA 隔离。

## 安装

1. 先将当前项目及数据另做安全备份；旧版本标签 `p2a-baseline-v1` 不要修改。
2. 把 ZIP 解压到独立目录，例如 `path/to/p2b_phase2_patch`。
3. 将其中的 `p2b_merge.py` 与 `p2b_validate_ir.py` 两个脚本复制到原实验工作目录 `path/to/code_marl_llm_slicing` 的根目录。
4. 在 PowerShell 执行以下命令。

## 步骤 1：只读预检

```powershell
conda activate marl
cd path/to/code_marl_llm_slicing

python .\p2b_merge.py `
  --holdout-manifest .\p1b_data\p1c_val_seed4601.json
```

不带 `--execute` 时不会新建经验库。应看到 `STATUS: read-only preflight passed`，并输出新旧数量、SHA 来源记录、六组构库规划和分层分布。

如果失败，先停下并把**错误文本**发给我，不要删除任何旧库，也不要强行放宽检查。

## 步骤 2：写入六个新库

只有步骤 1 通过后运行：

```powershell
python .\p2b_merge.py `
  --holdout-manifest .\p1b_data\p1c_val_seed4601.json `
  --execute
```

新库写入 `p2b_libraries/p2b_standard_300_v1` 等六个目录，另外生成 `p2b_libraries/p2b_composite_build_summary_v1.json`。原两个来源库完全只读，任何目标目录已存在都会拒绝覆盖。

## 步骤 3：原 P1-B 审计 + 重建每个新库的 IR

```powershell
python .\p2b_validate_ir.py `
  --holdout-manifest .\p1b_data\p1c_val_seed4601.json
```

这一命令：

1. 逐库调用项目自有的 `p1b_audit.py`；
2. 对每个库实际实例化原始 `LocalHistory`；
3. 调用原始 `p1d_build_ir.py` 为每个库生成**独立**历史 IR；
4. 实例化 `IRHistory` 检查其与当前对应角色库的兼容性。

IR 输出写入 `p2b_ir_libraries/p2b_standard_300_ir_v1` 等目录。任何同名目标已存在则停止，不会覆盖现有 IR 历史。

如果第 3 步因 `p1d/ir.py` 的输入约束失败，**不要直接修改 IR 校验逻辑**。把完整报错和 `p1d/ir.py` 发给我，按真实格式修订兼容层。

## 重要的可比性问题

- 两份原始库的 P1-B 经验已经通过各自的硬筛选；这里仅对合格经验进一步子集选择，不重新用旧开发场景调筛选阈值。
- 复合库 `training_sample_sha256` 的值是**两个源训练集来源签名的 SHA-256**，而不是某个单独训练样本文件的 SHA。`source_training_datasets` 保留两个原始数据集的独立 ID 和 SHA。
- 本工具同时检查两个原训练来源与传入验证集的来源 ID/SHA，但不同来源的哈希不代表已经穷尽排查所有跨集内容重复；更严格的场景级指纹排重需要完整原始场景文件。
- 旧的 P2-A 经验库、IR 历史、200 场景结果及代码标签均不应修改。
- 最终测试集必须在固定方案后评估。新增验证集的生成与 Qwen 运行放到下一步，不要在构库成功前运行。
- 不要把 `private_offline` 内含的全局联合结果或调试字段传入在线 Qwen Prompt。在线只使用对应角色的历史局部经验与其历史自体 IR 信息。

## 当前验证程度

- 在合成旧/新源库上，8 个离线测试通过。
- 合成生成的六份库已用你上传的原 `p1b_audit.py` 逐库验证通过。
- **当前没有拿到真实旧 152 条与新 1,209 条的实际记录。** 因此尚未在你的真数据上完成构库，也未运行真正的 `p1d/ir.py`、`LocalHistory` 或 Qwen。
- `p1d_build_ir.py` 只是入口；完整 IR 实现不在这次上传的 ZIP 中，故保留步骤 3 的严格本地验收。

## 本地自检（可选）

在补丁目录运行：

```powershell
python -m unittest discover -s tests -p "test_p2b_merge.py" -v
```

这只用合成库，不修改研究数据。
