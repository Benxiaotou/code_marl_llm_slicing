# P2-A：IR 风险感知的局部协作经验重排序

## 0. 本轮目标与边界

本轮仅增加一种方法：`ir_rerank`（P2-A risk-aware reranking）。

它在**同一个** `p1c_val_seed4601` 开发验证集（200 场景）、**同一个** `train_seed2601_v1` 训练历史库（152 条/角色）、同一个 Qwen、同一个 P1-D 完整 IR-aware Prompt 下运行。

P2-A **只更换检索到的历史示例**：首先使用 P1-C 局部相似度检索候选，然后以各角色**训练时历史的自身** IR 余量和调整成本进行风险重排序。执行时不读取完整网络状态、其他切片当前观测、真实当前收益，且不调用 Guard、LP 求解器或环境模拟来挑动作。Qwen 权重保持冻结。

本轮特意只改变 k1、k2、k3 的示例选择；g 的检索和 Prompt 与原 IR-aware 完全相同。

这是一项**开发验证实验**，200 个场景已被多轮分析过，不能视为论文最终未见测试集。

## 1. 代码安装与测试

前提：你已安装 P1-A～P1-E，并且具有：

- `p1b_libraries/train_seed2601_v1/`：152 条 MAPPO 合作历史案例/角色，P1-B 审计通过。
- `p1d_libraries/train_seed2601_ir_v1/`：由 P1-D 在**训练阶段**导出的自身历史 IR 信息，审计通过。
- `p1b_data/p1c_val_seed4601.json`：200 场景原验证集 manifest。
- 原 P1-C/P1-D/P1-E 的实验结果仍在已有目录下。

**先备份** `p1c_runner.py` 和 `p1c/policy.py`；然后把 ZIP 解压到：

`F:\Projects\code_marl_v0_exper_1`

此次只覆盖以上两个文件；新增 `p2a/`、`p2a_preview.py`、`p2a_compare.py`、`tests/test_p2a.py`、本 README 和发布 manifest。**不会覆盖** `main.py`、原 MAPPO checkpoint、原 P1-B/P1-D 经验数据及任何先前运行记录。此前 P1-C/P1-D/P1-E 的原方法仍然可用。

```powershell
conda activate marl
cd F:\Projects\code_marl_v0_exper_1

python -m unittest discover -s tests -p "test_p2a.py" -v
python -m unittest discover -s tests -p "test_*.py" -v
```

本补丁在离线模拟环境中完成 **65 项**测试，其中原 P1-A～P1-E 54 项回归，新 P2-A 11 项。真实内网 Qwen 和 GEANT 运行在你的本地机器进行，离线测试不等于真实方法性能。

## 2. P2-A 算法定义（v1 默认参数已预设）

对 k1、k2、k3：

1. 使用 P1-C 的训练库标准化**本角色局部数值距离**选出前 `pool_size=20` 条历史经验。
2. 永远固定最相似的一条作为**锚点**。
3. 对其余候选设置近邻门槛：`D <= min(D_min+0.10, 1.35*D_min)`。若门槛内不足 3 条，则保留原最近的 3 条作为保底候选。
4. 从 P1-D 的**本角色历史训练 IR overlay**生成风险百分位：

   `risk = 0.55 * cost_rank + 0.30 * (1 - relative_IR_margin_rank) + 0.15 * (1 - IR_margin_rank)`

   排名仅使用本角色 **152 条训练经验**统计。较高历史成本和较低历史 IR 余量，风险数值越大。
5. 在近邻范围内计算：

   `score = 0.65 * (D-D_min)/max(D_min,0.15) + 0.35 * risk`

6. 保留最近邻锚点；余下两条按 score 从低到高选择。将选出的 3 条完整原始示例（仍为**本角色历史状态、教师动作、本角色历史 IR 反馈**）传入原 P1-D IR-aware Prompt，由 Qwen 自主输出动作。

对 g：保持 P1-D 原始的最近 3 条经验，不进行风险重排序。

**重要限制：**历史 IR/成本是当时**联合动作**的实际结果，不是某个单独动作的因果收益或对当前场景的收益预测。风险数值只是历史案例选择的启发式指标。近邻门槛与锚点意在降低“低成本但状态不相似”的误选风险；不保证理论可行性或性能提升。

参数 `--risk-pool`、`--risk-weight`、`--risk-max-gap`、`--risk-max-ratio` 可用于以后做独立超参实验；**此次建议先只用默认参数运行一次**，不要在反复查看这 200 场景的表现后持续调参并把它们当最终结果。

## 3. 推荐先运行检索预览：不调用 Qwen，也不执行 env.step

```powershell
python p2a_preview.py `
  --library .\p1b_libraries\train_seed2601_v1 `
  --ir-history .\p1d_libraries\train_seed2601_ir_v1 `
  --dataset-manifest .\p1b_data\p1c_val_seed4601.json `
  --start 0 --count 200 --examples 3 `
  --output-dir .\p2a_runs\preview_val200_v1
```

生成：

- `p2a_runs/preview_val200_v1/preview.json`
- `p2a_runs/preview_val200_v1/retrieval_preview.csv`

其中 `changes_by_role` 表示 k1、k2、k3 分别有多少个场景与原始近邻 3 案例**不完全相同**；g 应严格为 0。`mean_eligible_by_role` 是每角色平均有多少条历史候选通过局部距离门槛。

如果 k1、k2、k3 的改变次数都接近零，那么直接用默认配置跑完整 800 次 Qwen 可能价值不大；建议先反馈 preview 结果，检查训练历史覆盖，而不是为追求指标而在这 200 场景上盲目调参。

**注意**：预览调用原环境的局部观测生成接口，但不执行资源分配动作、不做仿真结果评价，也不把环境中的全局对象传入检索算法。

## 4. 先进行 5 场景接口测试（使用真实 Qwen）

```powershell
python p1c_runner.py `
  --library .\p1b_libraries\train_seed2601_v1 `
  --ir-history .\p1d_libraries\train_seed2601_ir_v1 `
  --dataset-manifest .\p1b_data\p1c_val_seed4601.json `
  --method ir_rerank --examples 3 `
  --start 0 --count 5 --seed 1 `
  --model qwen3-next-80b-a3b-instruct `
  --temperature 0.1 --max-tokens 4096 `
  --api-timeout 120 --on-error abort `
  --output-dir .\p2a_runs\probe5_ir_rerank
```

共 5 × 4 = 20 次模型调用，仅用于接口验收。确保 `LLM_API_KEY`、`LLM_BASE_URL` 与以前 P1-D 相同。

## 5. 使用相同 200 个场景完成 P2-A 正式开发对照

```powershell
python p1c_runner.py `
  --library .\p1b_libraries\train_seed2601_v1 `
  --ir-history .\p1d_libraries\train_seed2601_ir_v1 `
  --dataset-manifest .\p1b_data\p1c_val_seed4601.json `
  --method ir_rerank --examples 3 `
  --start 0 --count 200 --seed 1 `
  --model qwen3-next-80b-a3b-instruct `
  --temperature 0.1 --max-tokens 4096 `
  --api-timeout 120 --on-error abort `
  --output-dir .\p2a_runs\val200_ir_rerank_v1
```

共有 200 场景 × 4 个角色 = **800 次模型调用**。P1-D 的完整 IR-aware 是最重要的对照：模型、基本 Prompt、训练 IR 来源和当前状态摘要完全一致，改变的只是 k1/k2/k3 的**历史示例选择**。

日志仍是原格式：`manifest.json`、`records.jsonl`、`metrics.csv`、`summary.json`。每角色新增 `p2a_rerank_trace_for_audit_only`，用于核查候选案例、选择得分和是否改变；这段审计信息**不会发给 Qwen**。

## 6. 与原八种方法同场景比较（推荐完整命令）

以前的八组结果不用重跑。使用新 `p2a_compare.py`（内部复用**不修改**的 P1-E 八方法比较器）：

```powershell
python p2a_compare.py `
  --knn .\p1c_runs\val200_knn `
  --icl .\p1c_runs\val200_icl3 `
  --ir-icl .\p1d_runs\val200_ir_icl3 `
  --ir-rerank .\p2a_runs\val200_ir_rerank_v1 `
  --ir-prompt-only .\p1e_runs\val200_ir_prompt_only `
  --ir-data-only .\p1e_runs\val200_ir_data_only `
  --mappo .\p1c_runs\val200_mappo `
  --random-icl .\p1c_runs\val200_random_icl3 `
  --zero .\p1c_runs\val200_zero `
  --bootstrap 5000 `
  --output-dir .\p2a_runs\compare_val200_all9
```

生成：

- `p2a_runs/compare_val200_all9/comparison.json`
- `p2a_runs/compare_val200_all9/paired_scenes.csv`

新增对照在 `pairwise_comparisons.ir_rerank_vs_ir_icl` 和 `ir_rerank_vs_mappo` 中；检索审计在 `p2a_retrieval_audit` 中。比较器将拒绝不同场景、模型参数、训练/IR 哈希、Prompt 模板、已有切片最近邻锚点或 g 检索不一致的情况。

如果只想先比关键三组，可省略 Random-ICL、Zero-shot、两个消融和 MAPPO 参数，但必须保留 `--knn`、`--icl`、`--ir-icl`、`--ir-rerank`。此时不能做完整九方法表。

**评价优先级：**合作有效率及其配对差异区间 → 系统效益 V → 调整成本 → 新切片接纳率 → 时延。也要报告改动率。如果效果没有改善，明确承认简单 IR 风险重排序可能过度偏向历史容易场景或不适合这种协作任务。

## 7. 后续如何扩充与均衡经验库

目前训练库规模为 152 条/角色。理论上，扩充到 300、500、1,000 条**高质量且覆盖不同状态的经验**，可能让同一当前局部状态找到更多可比较的安全动作，从而提高 P2-A 的候选池质量；但并不保证结果单调上升。

推荐的独立扩库研究：

1. 生成新**训练集**（不能混入现有 200 场景或未来正式测试集），运行**冻结 MAPPO**生成历史联合行动。
2. 与 P1-B 相同的合作有效、个体理性约束筛选，再按重叠节点/链路程度、资源压力、历史 IR 余量、调整成本等分层保留，避免全是简单场景。
3. 保持四个角色的经验仍对应同一个历史联合场景，再为各角色重建 P1-D 历史 IR overlay。
4. 把经验库大小作为**一个独立消融因子**，比较覆盖率、最近邻距离、重排序改动率，以及同场景 GEANT 效益。经验数量不同意味着训练信息预算不同，论文中要明确说明。

不要把低质量历史经验直接填满库；即使库变大，如果检索前 20 条变得同质、或离线历史结果与当前状态不匹配，效果也可能下降。

## 8. 科研限制与下一步

- 本轮仍使用已反复分析过的 `p1c_val_seed4601` **开发验证集**，所有性能变化只用于选择研究方向。
- 这轮不是微调，不是集中式测试时优化，不做执行前 Guard/LP/环境试运行。
- 不能把“高历史 IR 余量”当作当前场景一定安全的保证；它与历史其他三个智能体动作有关系。
- 为减少选择偏差，一旦确定检索配置，应在全新**独立**场景集上验证；最终论文还应考虑 Qwen 重复采样及相对 MAPPO 的时延开销。

完成后建议上传 `preview.json`、`compare_val200_all9/comparison.json`，以及 `val200_ir_rerank_v1` 结果 ZIP。优先看是否让当前完整 IR-aware 的 **74.5% 合作有效率**继续提升，同时保留较高的 **95% 接纳率**，而不是仅盯着接纳率。
