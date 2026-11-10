# P2-B：协作经验库扩充与质量实验（第一阶段）

## 目的与边界

这一工具包先执行**新增训练场景生成 → 冻结 MAPPO 教师经验采集 → 原 P1-B 硬筛选与审计**，并提供可复现的旧基线检索覆盖审计工具。

**本包不包含最终 152/300/500/1000 嵌套库的合并实现。** 原 P1-B 的库文件本身及 `p1b_select.py`、`p1d_build_ir.py` 的完整源码不在当前可核查材料中；不能把来自不同训练种子的两个库简单拼接后冒充已通过原读取器审计的库。因此先安全完成原始经验采集，待核对本地真实数据格式后再构建最终嵌套库与 IR 历史。

以下命令仅适用于你之前已经跑通的 Windows 项目：
`path/to/code_marl_llm_slicing`。建议先新建 `experiment/p2b-library-expansion` 分支，**不要更改** P2-A 标签指向的提交，也不要覆盖旧目录。

## 内容

- `p2b_stage1.ps1`：独立阶段的 Windows PowerShell 入口，拒绝覆盖已有数据；调用你本地已经存在的四个 P1-B 脚本。
- `p2b_coverage.py`：只读取 `records.jsonl` / `manifest.json` 的检索距离与收益，用于新旧经验库检索覆盖分析；也支持单运行 ZIP。
- `example_baseline_audit/coverage_summary.json`：根据你已上传的 P2-A 200 场景运行记录实际计算出的基线。
- `example_baseline_audit/coverage_per_scene.csv`：逐场景、逐角色距离与合作结果，无原始 Prompt 和内网地址。
- `tests/test_p2b_coverage.py`：3 个本地单元测试。
- `P2B_PHASE2_IMPLEMENTATION_SPEC.md`：构建真正公平的嵌套库及质量对照的验收标准。

## 运行前检查

在 PowerShell 执行：

```powershell
conda activate marl
cd path/to/code_marl_llm_slicing

# 将 ZIP 中的 p2b_stage1.ps1 和 p2b_coverage.py 放入此目录
Get-Item .\p1b_generate.py, .\p1b_collect.py, .\p1b_select.py, .\p1b_audit.py
Get-Item .\p1b_libraries\train_seed2601_v1
Get-Item .\p1d_libraries\train_seed2601_ir_v1
Get-Item .\p1b_data\p1c_val_seed4601.json
```

关键操作**只在原实验工作目录**执行，不要直接向发布仓库复制大量数据或内网配置。

## 第一步：生成新的 1,600 个训练候选场景

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\p2b_stage1.ps1 `
  -Stage generate -Seed 2602 -Count 1600
```

期望产生 `p1b_data/p2b_train_seed2602.json` 及配套数据。新种子 `2602` 不等于已有的训练种子 `2601` 或开发验证种子 `4601`。

如果目录已存在或有历史同名数据，脚本不会自动覆盖；应该先核对已有 manifest 的参数和 SHA，而不是删除结果直接重跑。

## 第二步：3 场景 MAPPO 接口验证

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\p2b_stage1.ps1 `
  -Stage probe -Seed 2602 -Count 1600
```

注意核查教师确实是 **live frozen MAPPO**、所有记录状态均为 OK，且没有 fallback、mock、固定动作替代或其他策略。

## 第三步：正式采集全部候选经验

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\p2b_stage1.ps1 `
  -Stage collect -Seed 2602 -Count 1600
```

输出目录为 `p2b_runs/p2b_train_seed2602_all1600`。

不要将先前使用过的 `p1c_val_seed4601` 200 个开发场景或以后最终测试场景混入教师经验。

## 第四步：原筛选规则先产出一个候选库

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\p2b_stage1.ps1 `
  -Stage select -Seed 2602 -Count 1600
```

输出 `p2b_libraries/p2b_train_seed2602_candidate_v1`。此处的 `--max-keep 1600` 仅表示“尽量保留所有满足既有规则的候选”，**不是最终实验库**。必须坚持已验证的联合合作条件，不能为了凑 1,000 条而把无效合作样本选进去。

## 第五步：审计新候选库与旧开发集隔离

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\p2b_stage1.ps1 `
  -Stage audit -Seed 2602 -Count 1600
```

至少核查 `audit_passed=true`、四角色数目和案例 ID 对齐、每角色只携带自身历史局部状态与动作、没有训练/验证场景泄漏。单靠随机种子不同不够，应当使用 manifest 来源哈希、原始场景指纹、case_id 交叉检查。

如果可用候选不足以做 1,000 条，追加新的训练批次及独立 seed，绝不降低 IR 硬筛选标准。

## 检索基线审计（现在就能复现）

```powershell
python .\p2b_coverage.py `
  --run .\p2a_runs\val200_ir_rerank_v1 `
  --output-dir .\p2b_runs\coverage_old152
```

完成阶段二的新库评估后，再运行：

```powershell
python .\p2b_coverage.py `
  --run .\p2a_runs\val200_ir_rerank_v1 `
  --compare .\p2b_runs\new_library_same_val200 `
  --output-dir .\p2b_runs\coverage_paired
```

第二条命令仅适用于新旧方法**确实在同一个数据集 SHA、同一批场景**上运行，且检索特征的距离标度可比较；如果缩放器在不同库上重新拟合，距离下降不可直接等同于覆盖改善。对研究结论，应优先比较新的独立验证集上的合作有效率、V、cost 和同场景配对置信区间。

## 下一阶段正式实验（完成嵌套库构建后再跑）

数据计划：

| 数据/场景 | 用途 |
|---|---|
| `p1b_train_seed2601` | 旧 152 条的来源，保持不变 |
| `p2b_train_seed2602` | 新增 1,600 个教师采集候选 |
| `p1c_val_seed4601` | 原有开发集，只作为回归对照，避免继续针对它调参 |
| `p2b_val_seed5601`，200 场景 | 新增开发验证，用来选择规模与质量策略 |
| `p2b_test_seed6601`，512 场景 | 最终独立测试，**固定方案前不要用于选择参数** |

在选定实验矩阵后再生成新验证集：

```powershell
python p1b_generate.py --name p2b_val_seed5601 --split validation --seed 5601 --count 200
```

建议研究矩阵：

1. **规模**：原 152 → 300 → 500 → 1,000，使用嵌套、训练来源对齐、选取规则固定的四角色联合案例。
2. **质量**：在相同规模（推荐 500）下比较常规筛选与压力/重叠/IR/多样性均衡筛选。
3. **方法**：第一轮先用 kNN 和离线距离检查；随后固定 Qwen Prompt + 3-shot，对 IR-aware 与 P2-A 重排序做同场景比较。
4. **指标**：合作有效率为主，V 与 cost 为重要副指标；同时提供 IR 违反率、接纳率、动作多样性、邻居距离与候选分布、配对 Bootstrap 区间。
5. **历史 IR**：每个新的 P1-B 角色库，都须再执行一次 `p1d_build_ir.py`，为本版本生成匹配的 `--ir-history`；严禁把旧 IR 文件直接复用给新库。

## 当前验证范围

`p2b_coverage.py` 已使用用户提供的真实 P2-A 200 场景 ZIP 完成计算，3 项合成数据测试通过。`p2b_stage1.ps1` 调用的本地项目脚本、MAPPO checkpoint 与 Windows PowerShell 环境无法在这里运行，所以**本工具包没有替用户实际生成新的 GEANT 场景，也没有完成扩库**。

请先完成前五步，再按 `P2B_PHASE2_IMPLEMENTATION_SPEC.md` 对接真实库格式。
