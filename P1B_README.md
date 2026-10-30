# P1-B：冻结 MAPPO 教师经验采集与高质量联合经验筛选

本包**只增加文件**，不修改 `main.py`、`p0_runner.py`、`p1a_runner.py`、原环境奖励、已有模型或历史实验数据。

## 1. 研究目标与边界

- 教师：已经训练完成、冻结的 MAPPO checkpoint。**不重新训练 MAPPO，不微调 Qwen，不调用内网 API**。
- 采集：每个场景采集同一步决策之前四个智能体各自的**原始局部 240 维观测**，教师 4×4 原环境动作（g 第 4 个环境位置无效），以及原环境 `step()` 的真实结果。
- 联合筛选：仅将 `g_done=1`，`V_ki >= V_ki_0`（三个已有切片），`V_g >=0` 的场景视为合作有效候选。
- 质量与覆盖：按资源竞争压力和重叠比例分层，**在每层内部**抑制最高成本样本，并综合 V、成本和状态多样性选择案例。阈值与权重必须只在训练/验证阶段确定。
- 分布式部署：导出四份角色库，每份**只含本角色历史局部观测及对应动作**。不输出其他角色的观测、联合动作、团队效益、全局状态或 Guard 优化解。
- 研究限制：**一次联合成功不代表每个局部动作因果最优**，因此 P1-C 必须与无需 LLM 的 kNN-Action 对照。

## 2. 文件内容

```
p1b_generate.py        使用原 GEANT 场景生成逻辑生成独立训练/验证/测试数据
p1b_collect.py         冻结 MAPPO 教师采集与原环境重评估，支持断点续跑
p1b_select.py          合作硬筛选 + 分层成本/V/覆盖选择 + 四份局部经验
p1b_audit.py           各角色经验无全局泄漏、动作正确、训练/测试隔离审计
p1b/core.py            240 维观测/动作/指标/来源严格校验
p1b/teacher.py         复用本地 P0 MAPPOPolicy 或重放已有 MAPPO 日志
p1b/select.py          资源压力特征与分层选择算法
tests/test_p1b.py       采集/筛选/回读/隔离测试
tests/test_p1b_generation.py  新数据生成安全测试
```

必须已经有可运行的 P1-A (`p1a_runner.py` + `p1a/`) 和 P0 (`p0_runner.py` + 已训练 MAPPO checkpoint)，`main.py`、`configs_main.yaml` 和 GEANT 拓扑/流量数据。使用你已验证的 `(marl)` Conda 环境。

> 注意：由于当前交付环境不能加载你 Windows 的 XuanCe checkpoint 或访问 GB 级完整场景，新文件已通过**模拟环境端到端测试和 P1-A 回归测试**，但**真实冻结 MAPPO + 新数据的本地联调还需运行下述 1 场景验收**。若本地 `p0_runner.MAPPOPolicy` 的返回字段与预期不同，程序将报告确切字段并停止，绝不会生成伪造的教师动作。

## 3. 安装与离线测试

将 ZIP 解压至项目根目录，例如：

```powershell
conda activate marl
cd F:\Projects\code_marl_v0_exper_1
python -m unittest discover -s tests -p "test_p1b*.py" -v
```

## 4. 先用已有数据做**1 场景接口验收**（开发用途，不能当作训练集）

先用已有的原 600 场景测试 MAPPO bridge 是否与当前 `p0_runner.py` 兼容：

```powershell
python p1b_collect.py `
  --dataset legacy --allow-used-dev-dataset `
  --teacher live --start 0 --count 1 --seed 1 `
  --output-dir .\p1b_runs\legacy_probe1 --on-error abort
```

这会调用冻结 MAPPO，不调用 Qwen，也不训练。应显示 `scene=0 accept=... IR=... V=... cost=...`，并生成 `joint_records.jsonl`。如失败，请查看 `joint_records.jsonl` 里的 `failure_message`；不要直接扩大到 1000 场景。

`--teacher-space` 默认是 MAPPO 的 **[-1,1] 环境动作**。只有当你的原 P0 输出明确定义为 `[0,1]` 比例时，才设置 `--teacher-space ratio`。**程序不会凭正数自动猜测动作空间。**

### 可选：重放已有 P0 MAPPO 日志（仍然仅用于开发审计）

```powershell
python p1b_collect.py `
  --dataset legacy --allow-used-dev-dataset `
  --teacher recorded --teacher-run .\p0_runs\mappo_paper512_fix `
  --start 0 --count 10 --output-dir .\p1b_runs\legacy_replay10
```

需要该目录包含 `mappo_records.jsonl` 和证明 `policy=mappo` 的 `summary.json` 或 `manifest.json`，且每条记录含有**实际 MAPPO 动作**。如果旧日志只有 V/接纳而没有动作，程序会拒绝；此时使用上面的 `--teacher live`。重放记录将标记为 `recorded_MAPPO_replay`，不是实时模型推理。不允许使用 Qwen、Guard、固定策略日志冒充 MAPPO。

已有原 600 场景及 512–591 已用于 P0/P1-A 调试：不要把它们用于论文最后的独立测试，更不要把历史测试的结果当作未见训练样本。

## 5. 生成一个**新独立**训练集（先 200，后 1000）

先用新种子生成 200 个训练场景，用于检查真实模型与数据生成逻辑：

```powershell
python p1b_generate.py `
  --name p1b_train_seed2601 --split train --seed 2601 --count 200
```

它会运行原来的 `CustomMultiAgentEnv` 样本生成分支，在 `data/GEANT/p1b_train_seed2601.pkl` 创建新数据，并记录 `p1b_data/p1b_train_seed2601.json`。**不会覆盖原来 1.8 GB 的 600 场景文件，也不改原代码。** 源程序是整体 pickle 序列化，200 场景仍可能需要数百 MB 磁盘与较大 RAM，只运行可信项目 pickle。

先对新数据做一次真实教师单场景验收：

```powershell
python p1b_collect.py `
  --dataset custom --dataset-manifest p1b_data/p1b_train_seed2601.json `
  --teacher live --start 0 --count 1 --seed 1 `
  --output-dir .\p1b_runs\train_seed2601_smoke1 --on-error abort
```

通过后，在**新目录**采集全部 200 个（**不要**用单场景目录 `--resume` 扩展 count，因场景清单改变会被拒绝）：

```powershell
python p1b_collect.py `
  --dataset custom --dataset-manifest p1b_data/p1b_train_seed2601.json `
  --teacher live --start 0 --count 200 --seed 1 `
  --output-dir .\p1b_runs\train_seed2601_full200 --on-error abort
```

同一个**200 场景目录**如果出现意外中断，可使用完全相同参数加 `--resume`。已记录的场景不会再次运行；改变 seed/模型/场景清单会被拒绝。如果先前某场景已经记录为失败，建议单独调查，不要把失败场景当作正例。

稳定后，按需生成独立的 1000 场景训练数据（用**新的 ID 和种子**），同时生成不同种子的验证集和最终测试集：

```powershell
python p1b_generate.py --name p1b_train_seed3601 --split train --seed 3601 --count 1000
python p1b_generate.py --name p1b_val_seed4601 --split validation --seed 4601 --count 100
python p1b_generate.py --name p1b_test_seed5601 --split test --seed 5601 --count 512
```

**最终测试集严格封存**，不得用于筛选示例、调提示词或选择方法。原 MAPPO checkpoint 也必须保持冻结。上述数量均为实验建议，可先完成 200 案例的工程流程后再扩大。不要使用不同种子生成但随后人工混合或重复的样本。

## 6. 运行联合经验筛选与导出

```powershell
python p1b_select.py `
  --collection .\p1b_runs\train_seed2601_full200 `
  --output-dir .\p1b_libraries\train_seed2601_v1 `
  --max-keep 160 --cost-quantile 0.95
```

其中，`--cost-quantile 0.95` 是每个资源压力/重叠层中的相对高成本过滤，而不是统一的绝对成本上限。`--max-keep 160` 仅为初始示范，并非论文固定超参数。

输入 `joint_records.jsonl` 仅供离线筛选，输出目录包含：

```
p1b_libraries/train_seed2601_v1/
  library_manifest.json
  selection_report.json
  selection_report.txt
  audit_report.json               # 审计之后产生
  private_offline/
    selected_joint_records.jsonl  # 仅供离线，不允许进入 Qwen Prompt
    selection_index.json          # 仅供离线
  role_libraries/
    agent_k1.jsonl
    agent_k2.jsonl
    agent_k3.jsonl
    agent_g.jsonl
```

每条 `agent_k*.jsonl` **仅**包含原始 `observation_240`、本角色 `own_teacher_ratios`、`own_teacher_env_action`、角色/来源 ID。`agent_g` 仅保留三个有效动作。**不含四角色联合状态、其他动作、效益或来源评分。**

### 经验库隔离审计

```powershell
python p1b_audit.py `
  --library .\p1b_libraries\train_seed2601_v1
```

验证/测试集生成后，可增加防泄漏检查：

```powershell
python p1b_audit.py `
  --library .\p1b_libraries\train_seed2601_v1 `
  --holdout-manifest .\p1b_data\p1b_val_seed4601.json `
  --holdout-manifest .\p1b_data\p1b_test_seed5601.json
```

这会拒绝相同的 `dataset_id` 或相同样本文件 SHA-256。另请人工保留各生成批次的随机种子和可复现环境版本，不能仅靠散列证明统计独立性。

## 7. 如何解释筛选指标

硬门槛：`g_done=1`、`V_k1>=V_k1_0`、`V_k2>=V_k2_0`、`V_k3>=V_k3_0`、`V_g>=0`。

然后使用**原四份局部观测在离线阶段聚合**的资源压力特征（g 本地原观测中的已分配/请求比例，k 本地重叠比例与利用率）建立 3 个压力层 × 2 个拓扑重叠层。每层内部使用效益排名与反向成本排名生成综合质量分，辅以状态多样性进行轮转采样。

这是一套**可实验验证的初始筛选规则**，不能预先称为最优信用分配算法。尤其不能把“联合成功”误认为单个切片的动作在独立意义上正确。下一步 P1-C 应分别比较：MAPPO、kNN-Action、Qwen-Random-ICL、Qwen-Coop-ICL，以及 0/1/3/5 案例数的消融。

## 8. 兼容性与诚实边界

1. **本地 P0 接口**：基于你前面成功运行的 `p0_runner.MAPPOPolicy(config, checkpoint).act(obs)`。这部分无法在交付环境中调用本地 `.pth` 和 XuanCe，因此第一步的 `legacy_probe1` 是必要验收，而不是已验证的真实运行结果。
2. **旧环境遗留规则**：g 计算/缓存压缩与准入检查可能存在不一致。本包不修改原环境；如后续修复，MAPPO 和所有 Qwen 实验要统一重评估。
3. **原环境数据量**：1000 场景可能占数 GB，采集时 P0 内部 MAPPO 策略初始化可能再次加载环境，请先用 1 场景和 200 场景测试内存。不要一开始创建多个 1000 场景环境实例。
4. **严格区分来源**：`live_frozen_mappo` 为实时冻结教师动作；`recorded_MAPPO_replay` 为历史教师动作回放；`development_only` 不可声称最终独立测试。
5. **训练/推理边界**：集中式联合奖励只参与**离线经验筛选**。在 P1-C 的真实执行时，每个 Qwen 只能检索自己的历史局部经验，并使用自己的当前观测决定动作。

## 9. 回滚

删除 `p1b/`、`p1b_generate.py`、`p1b_collect.py`、`p1b_select.py`、`p1b_audit.py`、`tests/test_p1b*.py` 及新输出目录即可。原有模型、P0/P1-A、GEANT 600 样本不受影响。
