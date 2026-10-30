# P1-C：局部数值检索 + kNN-Action + 冻结 Qwen 协作经验 ICL

## 一、实验目的与严格边界

**当前已有结果**：P1-B 通过审计，各角色含 152 条从冻结 MAPPO 产生、经联合合作有效性筛选的历史经验。**P1-C 不训练任何模型、不访问全局当前网络信息、不通信、不使用 Guard、LP 或测试时试运行来修正动作。**

新代码采用原 P1-A 的 `load_scene_local_observations()`、`run_one_scene()`、GEANT `env.step()` 和完整的系统效益/成本/合作有效性评价。仅向每个角色决策器提供**该角色当前原始 240 维局部观测**和**同一角色的历史局部案例**。

不同角色检索到的最近历史联合场景可以不同，因此不能保证最后联合动作协调；这正是本次要测的研究假设。

### 两条主要路线

| 方法 | 决策 | 是否调用大模型 | 历史示例 | 必要说明 |
|---|---|---|---|---|
| `knn` | 按本角色局部状态距离取最相似的一条 MAPPO 教师动作 | 否 | 1 | 强对照；4 个角色各自检索 |
| `icl` | 把本角色当前状态与 3 条最相似的教师历史案例交给冻结 Qwen；模型输出新动作 | 是 | 3（可调） | 核心实验 |

### 三个补充对照

| 方法 | 用途 |
|---|---|
| `random_icl` | 固定随机种子选 3 条本角色历史案例，检验局部相似检索本身的作用 |
| `zero` | 同一新验证集上运行原 P1-A Zero-shot Prompt，检验 ICL 相比零样本是否改善 |
| `mappo` | 同一新验证集运行原冻结 MAPPO checkpoint，得到公平的教师参考性能 |

**不要**用旧 80 个场景的 P1-A 数值与新验证集的 P1-C 结果直接做配对差异比较；需要在相同数据和同一环境版本重新测量（`zero` 命令已提供）。

## 二、新增文件（只追加，不覆盖旧代码）

```
p1c_runner.py               新验证/测试场景评估；knn、icl、random_icl、zero、mappo
p1c_compare.py              同场景配对对照、Bootstrap 区间、动作多样性
p1c/__init__.py
p1c/retrieval.py             角色独立历史经验、特征、局部相似距离
p1c/policy.py                kNN、冻结 Qwen ICL、Zero-shot 与 MAPPO 适配器
tests/test_p1c.py           信息隔离、数据划分、动作、配对、端到端模拟回归
P1C_README.md               本文件
```

依赖你已安装并通过审核的 P1-A 和 P1-B：`p1a_runner.py`、`p1a/`、`p1b/`、`p1b_generate.py`、`p1b_collect.py`、`p1b_audit.py`、`p0_runner.py`（只有运行 MAPPO 对照时需要）。

## 三、先验证 P1-B 经验库

在本地项目根目录执行：

```powershell
conda activate marl
cd F:\Projects\code_marl_v0_exper_1
python -m unittest discover -s tests -p "test_p1c.py" -v
python p1b_audit.py --library .\p1b_libraries\train_seed2601_v1
```

检查 `n_per_role=152`，`audit_passed=true`。不要将 `private_offline/selected_joint_records.jsonl` 提供给 Qwen；P1-C 运行时**根本不打开**这个目录，`p1c/retrieval.py` 只读取 `role_libraries/agent_*.jsonl`。

## 四、准备一批全新验证场景（建议 200 个）

本次建议比以前的 80 个开发场景再多一些，先用 200 个新生成的 GEANT **validation** 场景。不要把以前用于调参的 600 样本、原来 `512–591` 或经验库的训练样本作为新的验证/测试案例。

```powershell
python p1b_generate.py `
  --name p1c_val_seed4601 `
  --split validation --seed 4601 --count 200
```

生成：`p1b_data/p1c_val_seed4601.json` 和 `data/GEANT/p1c_val_seed4601.pkl`。这个生成过程运行原 GEANT 样本生成逻辑，不修改原环境。请确保磁盘和内存充足。

**如果你已经有另一批未用于经验筛选/调参的 validation 样本**，可以直接把相应 `p1b_data/<name>.json` 指定给 `--dataset-manifest`；不必生成重复数据。只允许 `validation` / `test`，不允许 `train`。未来最终测试仍需一套**从未用于 Prompt、示例数量或检索参数选择**的独立场景。

建议再次审计数据来源：

```powershell
python p1b_audit.py `
  --library .\p1b_libraries\train_seed2601_v1 `
  --holdout-manifest .\p1b_data\p1c_val_seed4601.json
```

P1-C 在运行时还会自动检查 train/validation ID 与样本哈希不重合，检查原 `main.py` 与配置文件 SHA-256、当前样本 pickle SHA-256。哈希不同并不能独自证明随机样本统计独立，生成种子及训练/验证划分仍需严格管理。

## 五、先运行 1 场景接口验证

### 1. kNN：不调用 Qwen

```powershell
python p1c_runner.py `
  --library .\p1b_libraries\train_seed2601_v1 `
  --dataset-manifest .\p1b_data\p1c_val_seed4601.json `
  --method knn --start 0 --count 1 --seed 1 `
  --output-dir .\p1c_runs\knn_probe1
```

这里的输出是真实 GEANT 环境对 kNN 历史动作的评价（如果在你的 Windows 项目上成功运行），无需 API。

### 2. Qwen-ICL 接口模拟（显式标注 Mock，不能作实验成绩）

```powershell
python p1c_runner.py `
  --library .\p1b_libraries\train_seed2601_v1 `
  --dataset-manifest .\p1b_data\p1c_val_seed4601.json `
  --method icl --examples 3 --mock-llm --start 0 --count 1 `
  --output-dir .\p1c_runs\icl_mock_probe1
```

### 3. 内网 Qwen 真实单场景联调

先确保已设置 `LLM_API_KEY` 与正确的 `LLM_BASE_URL`；本程序不会在日志中打印 API Key。默认为已用过的 `qwen3-next-80b-a3b-instruct`，可通过 `--model` 修改。

```powershell
python p1c_runner.py `
  --library .\p1b_libraries\train_seed2601_v1 `
  --dataset-manifest .\p1b_data\p1c_val_seed4601.json `
  --method icl --examples 3 --start 0 --count 1 --seed 1 `
  --model qwen3-next-80b-a3b-instruct --api-timeout 120 `
  --on-error abort --output-dir .\p1c_runs\icl_probe1
```

每个角色都必须返回合规 `{"ratios":[...]}` 数字；k1/k2/k3 为 4 维，g 为 3 维。程序**不会裁剪或修正**无效的 LLM 动作；失败直接报告，不拿固定动作作为替代。

## 六、正式运行 200 场景的对照实验

PowerShell 中可以先声明共用参数（减少误用不同场景/seed）：

```powershell
$common = @(
  '--library', '.\p1b_libraries\train_seed2601_v1',
  '--dataset-manifest', '.\p1b_data\p1c_val_seed4601.json',
  '--start', '0', '--count', '200', '--seed', '1',
  '--on-error', 'abort'
)
```

首先跑两条核心方法：

```powershell
python p1c_runner.py @common --method knn `
  --output-dir .\p1c_runs\val200_knn

python p1c_runner.py @common --method icl --examples 3 `
  --model qwen3-next-80b-a3b-instruct `
  --temperature 0.1 --max-tokens 4096 --api-timeout 120 `
  --output-dir .\p1c_runs\val200_icl3
```

再跑建议的补充对照（先有核心结果也可进行比较）：

```powershell
python p1c_runner.py @common --method random_icl --examples 3 `
  --model qwen3-next-80b-a3b-instruct --temperature 0.1 `
  --max-tokens 4096 --api-timeout 120 `
  --output-dir .\p1c_runs\val200_random_icl3

python p1c_runner.py @common --method zero `
  --model qwen3-next-80b-a3b-instruct --temperature 0.1 `
  --max-tokens 4096 --api-timeout 120 `
  --output-dir .\p1c_runs\val200_zero

python p1c_runner.py @common --method mappo `
  --checkpoint marl_models/mappo/dir_model_1/512000_GEANT_mappo_model_save.pth `
  --output-dir .\p1c_runs\val200_mappo
```

如果 `mappo` 加载 checkpoint 遇到兼容性问题，不要把 `knn` 经验动作伪装成教师基线，先检查 P0 接口再单独重跑。初版至少执行 `knn + icl`，最好完成全部五组。

**不同方法必须使用同一个数据集 SHA-256、同一场景列表、相同 seed、原环境源码和奖励实现。** 当某个方法 `policy_error` 或其他失败时，不得比较只有成功记录的均值；修复实际问题后用新输出目录完成整个实验。该版本故意禁止覆盖结果目录，没有自动续跑，避免重复场景的评估混淆。

## 七、自动配对比较

只有核心两组时：

```powershell
python p1c_compare.py `
  --knn .\p1c_runs\val200_knn `
  --icl .\p1c_runs\val200_icl3 `
  --output-dir .\p1c_runs\compare_val200_core
```

完成补充对照后：

```powershell
python p1c_compare.py `
  --knn .\p1c_runs\val200_knn `
  --icl .\p1c_runs\val200_icl3 `
  --random-icl .\p1c_runs\val200_random_icl3 `
  --zero .\p1c_runs\val200_zero `
  --mappo .\p1c_runs\val200_mappo `
  --bootstrap 3000 `
  --output-dir .\p1c_runs\compare_val200_all
```

输出：

```
manifest.json        （各 run）数据来源与算法、环境和经验库文件指纹
records.jsonl        （各 run）每场景完整动作/检索 ID/Prompt 哈希/环境指标
metrics.csv          （各 run）便于 Excel 读取的逐场景结果
summary.json         （各 run）完整尝试、接纳与合作有效汇总
comparison.json      多方法配对均值/差异/Bootstrap 95% 区间
paired_scenes.csv    每场景各方法接纳、合作有效、V、V_g、成本、时延
```

主要判据：合作有效率优先，其次是接纳率、整体效益 V、调整成本、各切片效益和延迟。还记录动作多样性、`ICL-only` / `kNN-only` 接纳场景数。配对 Bootstrap 假设场景独立，但不包含模型重复采样不确定性；此处 200 个场景仅为**开发验证**，不是最终 512 新场景的论文结论。

## 八、检索原理和研究解释

- 所有角色独立检索，不向其他角色共享当前状态；不统一检索“相同 joint case”。
- k1/k2/k3 的特征包含两个**连续节点控制组**、两个**连续链路控制组**的重叠数量、容量、使用率及静态风险参考，以及本切片总体压力。
- g 的特征包含计算/缓存/带宽请求与已分配比例、局部资源缺口以及局部拓扑摘要。
- 训练阶段对 152 条 role-local 历史特征拟合每维标准差缩放，设置 0.12 数值地板；运行阶段不拿测试样本重新拟合标准化统计。
- kNN 使用同一个本角色距离检索最近一条，并直接使用该教师动作；ICL 用最近 3 条历史动作作为示范，由 Qwen 自主生成连续动作。
- 历史 Prompt 包含本角色的数值摘要和本角色教师动作；当前 Prompt 包含本角色原始观察及对应数值摘要。**不暴露历史其他切片动作或联合效益数值**。
- 历史例子经过全局合作结果筛选，但 **联合成功不意味着单个历史动作独立最优**。
- 历史 Prompt 不展示静态无迁移参考阈值，以避免 P1-A+ 已发现的阈值/0.5 数值锚定问题；这些局部特征仍用于检索。

## 九、当前验证程度与风险

1. 随包离线单元和模拟环境集成测试已通过；没有内网模型服务、真实 Windows GEANT 原始大场景文件或你本地 152 条 P1-B 经验库，所以**交付环境并没有真实跑出任何 P1-C 性能结果**。
2. 程序复用本地原环境和历史接口，不改变奖励、迁移、重路由及补偿计算。
3. 原环境 g 的计算/缓存压缩和准入检查可能存在实现不一致，P1-C 为保证一致性不私自修改；如果今后统一修复，必须全部方法重新评价。
4. `p1b_audit.py` 检查防泄漏；P1-C 在运行时检查文件 SHA 与数据划分，但不同生成种子的独立性还需要人工保证，特别注意数据内容是否来自同一生成器副本。
5. 不能以高接纳率单独认定策略有效。原 LP-only 已出现高接纳、低合作有效的情况。
6. 如 ICL 仍低于 kNN 或 Zero-shot，不能把联合经验筛选产生的收益说成“大模型独立优化能力”。

请将新运行目录中的 `summary.json`、`manifest.json`、`records.jsonl`、`comparison.json` 和 `paired_scenes.csv` 打包上传，便于分析**哪些切片动作造成高成本、是否突破 0% 合作有效率，以及 ICL 是否真正超过 kNN**。
