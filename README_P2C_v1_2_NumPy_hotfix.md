# P2-C v1.2：MAPPO 数组路径比较兼容性修复

## 发现的问题

测试数据已经全部准备完毕（3 × 512 = 1,536 场景），但第一段 MAPPO 在 `main.py` 的 `_get_k_link_cost` 中出错：

```python
if k_net_new_shortest_paths[idx_flow] != k_net.shortest_paths_node[idx_flow]:
```

当两条路径为 NumPy 数组时，`!=` 返回一个布尔数组而非单一布尔值。修复后的判断为：

```python
if not np.array_equal(k_net_new_shortest_paths[idx_flow], k_net.shortest_paths_node[idx_flow]):
```

**绝不能直接修改一行后继续使用旧版队列。** 原 `p1c_runner.py` 和旧版 P2-C 队列都校验生成时 `main.py` 的 SHA256，直接修改会导致新的源代码与数据生成记录不匹配。本修复包用独立的评价适配器解决这一点，完整保留生成时的源码与哈希。

## 包内文件

| 文件 | 用途 |
|---|---|
| `p2c_qwen8_queue.py` | v1.2 队列：沿用同一份 1,536 场景、8 方法、384 任务，识别已审计评价补丁 |
| `p2c_eval_runner.py` | 评价专用适配器，保持旧 `p1c_runner.py` 不变；校验原生成源码与修复后的评价源码 |
| `p2c_apply_numpy_hotfix.py` | 一次性安全迁移：预检查、单行源码修复、存档旧协议锁和 0/0 失败记录 |
| `run_p2c_qwen8.ps1` | 与上一版相同的 PowerShell 启动器 |

## Windows PowerShell 使用步骤

先确认原来的 `-Mode all` 已经退出；不要两个队列同时运行。

1. 把 ZIP 解压进 **`path/to/code_marl_llm_slicing`** 项目根目录（不是 GitHub 发布目录）。用新版 `p2c_qwen8_queue.py` 替换旧队列文件。若希望保留旧队列，可先将它复制到根目录外的备份目录。
2. 在已激活的 `marl` 环境中执行：

```powershell
cd path/to/code_marl_llm_slicing

# A. 只读预检查：验证旧版主环境、旧协议锁、三套测试集，以及 0 个已评分场景
python .\p2c_apply_numpy_hotfix.py --check

# B. 真正修复：只修改已定位的一处 main.py 表达式，并归档旧状态
python .\p2c_apply_numpy_hotfix.py --apply

# C. 验证程序能识别已有三个数据集
python .\p2c_qwen8_queue.py --mode plan

# D. 继续运行：这一次会复用全部已有测试数据集，不进行重新生成
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\run_p2c_qwen8.ps1 -Mode all
```

先确认终端出现 `REUSE VERIFIED`（三个种子），然后关注 `[1/384]` 的 MAPPO 分段，期望出现 `PASSED`。若再次出现新的 `Traceback`，立即停止于失败分段并保存日志，不要擅自跳过。

**如果 A 步失败，就不要执行 B 步。** 将 `HOTFIX REFUSED` 的完整错误输出提供给分析人员。预检查会拒绝任何已经有正式评分、模型配置改变、场景损坏、原源码不匹配、原始锁不匹配的情况。

## 修复后的溯源数据

修复操作不会修改任何 `p1b_data/p2c_test_seed*.json`、`data/GEANT/p2c_test_seed*.pkl`、经验库、模型权重或配置文件。

原始记录会保存在：

```text
p2c_runs/qwen8_test_v1/provenance/
    main_generation.py                    # 生成 1536 场景时使用的完整环境源码
    protocol_lock_before_eval_hotfix.json # v1.1 旧版协议锁
    eval_hotfix_attestation.json          # 修复前后 main.py SHA256 及测试数据记录
    failed_attempt_mappo_6601_chunk_0000_032/
                                         # 首个未评分失败段的全部原始文件
```

修复后新的 `p2c_runs/qwen8_test_v1/protocol_lock.json` 会由新版队列在首次运行时重新创建，它绑定 3 个测试数据文件、生成数据原始源码、评价源码、评价适配器、经验库、模型及完整实验参数。此版本号为 `p2c_qwen8_serial_v1_2_numpy_path_hotfix`。

**论文可复现性说明**：测试场景是在原版环境代码下生成的；评价时仅修复 NumPy 最短路径数组相等性判断，原始生成数据和其 SHA256 保持不变。MAPPO 和所有 Qwen 方法将在同一修复环境下评价。此修复发生在第一次正式场景获得评分之前，不涉及择优修改算法或测试种子。

## 本包的已验证范围与局限

已通过 Python 语法检查，以及模拟环境下的完整性检查、仅一行源代码替换、旧协议与失败目录归档、数据集原始哈希保留、新队列数据验证。

**未在你本地真实 GEANT 环境与 MAPPO checkpoint 下执行**。因此不能保证修复完第一处后再不会遇到其他旧环境的 NumPy 比较错误。如有第二处报错，应继续按完整 traceback 定位，不要用全局替换 `!=` 的方式修复整个环境。
