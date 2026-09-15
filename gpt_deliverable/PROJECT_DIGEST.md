# GEANT 虚拟业务网络资源分配 · MARL 项目《GPT 全解说明书》

> **本文档用途**: 让一个**没有该项目文件**的 LLM(如 GPT)仅凭本文件 + 少量核心源码,即可完整理解本项目的
> **问题建模、环境动力学、奖励设计、训练流程、代码结构与实验家族**,从而指导后续 "用大模型替换项目某环节" 的论文开发。
>
> **文档目标读者**: GPT(你直接把本文交给它)。
> **配套文件**(建议一并提供): `main.py`、`configs_main.yaml` / `configs_main_INDRL.yaml` / `configs_main_SingleRL.yaml`、可选 `utlis/network_env.py`。
> **明确不提供**: `data/`(数 GB 的 XML/流量矩阵/样本 pkl),本文件 §6 已说明其格式,可运行时代码自动再生成。

---

## 0. 开发者原始目标(请 GPT 围绕此目标提供开发/论文规划)

后续研究目标是:**以本项目为基础,用"大模型(LLM)"替换其中某一部分工作/模块,并据此撰写一篇论文。**
请 GPT 结合本说明书先做两件事:
1. 确认对项目机制理解无误(尤其 §4 的奖励/代价/贡献度公式与 §7 的实验家族);
2. 基于 §9 的"候选替换点"给出具体的 LLM 介入方案、可行性、对比协议与论文定位建议。

---

## 1. 一句话定位

这是一个**基于多智能体强化学习(MARL)的"物理网络上多个虚拟业务网络(VN)共存时的资源让渡/网络切片"研究项目**。
物理网络使用真实 **GEANT 骨干网拓扑 + 2005 年真实流量矩阵(TM)**;其上已部署 3 个"已有业务网络 k1/k2/k3",
现在要**新建 1 个业务网络 g**。因为物理资源有限,g 需要 k1/k2/k3 在**重叠节点/链路**上"让出"资源;
MARL 算法(**MAPPO**)学习"各网络如何协调让出、g 如何调整自身需求",新网络 g 的收益按**贡献度(类 Shapley 分配)**
在 k1/k2/k3/g 之间分成,形成合作博弈式多智能体决策问题。

对比基线: 单智能体 RL(SingleRL)、独立 PPO(INDRL)、SPF 静态路由、DRL-TE/ScaleDRL 等(见 §7)。
项目是一系列论文实验代码的主干,含大量 `main_*.py` 变体文件。

---

## 2. 技术栈与运行环境

| 项 | 值 |
|---|---|
| 语言/生态 | Python 3.x, `torch==1.13.0`, `numpy==1.22.4`, `networkx==3.1`, `pandas`, `scipy` |
| 强化学习框架 | **XuanCe ==1.2.2**(开源 MARL 框架,含 `MAPPO`/`IPPO` 算法、`make_envs`、`RawMultiAgentEnv` 基类) |
| 算法 | MAPPO(`MAPPO_Agents`,Clip learner); 对比: IPPO/独立 RL、单智能体 PPO |
| 计算设备 | 默认 `cpu`(yaml 中 `device: "cpu"`) |
| 关键依赖代码 | `xuance`、`gym`、`networkx`,绘图 `matplotlib` |
| 数据 | GEANT 2005-04 第一周(训练 672 个 TM XML)+ 第二周(测试); `topology-anonymised.xml` |

> 注意: `requirements.txt` 是从一个更大的老环境导出的(含 tensorflow/keras 等无关包),实际 MARL 主流程只用 torch+xuance。

---

## 3. 系统全貌: 目录地图

```
code_marl_v0_exper_1/
├── main.py                      ★ 主入口: CustomMultiAgentEnv(MARL 环境)+ CustomMAPPO_Agents(训练/测试)
├── configs_main.yaml            ★ MAPPO 超参 + 路径 + GEANT 数据文件名
├── configs_main_INDRL.yaml     独立多智能体 RL(IPPO)配置
├── configs_main_SingleRL.yaml  单智能体 RL 配置
├── main_SingleRL.py / main_INDRL.py / main_ScaleDRL.py / main_SPF.py / main_DRL-TE.py …   §7 实验变体
├── utlis/
│   ├── network_env.py        ★ 拓扑解析(GEANT/Abilene/Xspedius/Uunet/SwitchL3/Dfn/Cernet/Chinanet/BtNorthAmerica)
│   │                        + 流量矩阵解析 + ReGEANTTopology(删点生成子拓扑)
│   ├── custom_marl_env.py    main.py 中 MARL 环境的**独立副本**(超大但内容与 main.py 基本一致)
│   ├── custom_environment*.py  CFR-RL/流量工程基线环境(drlte/scaledrl/spf/save/…), 及 _r1sc 可扩展版本
│   ├── config.py            CFR-RL 风格基线环境用的 Python 常量配置(经典 CFR-RL / DRL-TE 框架, 见 §8)
│   ├── general_function.py   setlogger / setup_seed / MAML 工具(早期论文遗留)
│   ├── game.py / model.py / models.py / train.py / *_generator.py  基线与早期 MAML 重放/微调工具
├── data/                     ★★ 数据(几 GB, 不随本说明提供, 见 §6): 拓扑 XML / 每周 TM XML / *_sample_save.pkl / TM1/TM2
├── result/  logs/  figure/  marl_models/  运行产物(模型 .pth、奖励/指标 pickle、日志、图)
└── docs/  F_save/  logger_save/  etc.      历史实验输出
```

---

## 4. MARL 环境详解(以 `main.py` 中的 `CustomMultiAgentEnv` 为准)

> `utlis/custom_marl_env.py` 是把这份环境单独抽出的副本;改环境以 `main.py` 为准。

### 4.1 实体与容量设定

**物理网络(GEANT)**:
- 23 个节点(索引 0..22), 有向链路段(每跳容量模板 10,000,000 kbps; `main.py` 会将其改写为 **33 Gbps**);
- 节点资源: `comp_capacity=1000`(计算)、`cach_capacity=500`(缓存);链路 `bandwidth=33 Gbps`;
- 负载来源: 真实 TM(kbps, `traffic_matrices`, GEANT 原始单位 kbps)。

**虚拟业务网络(每个 VN 一个 `networkx.DiGraph`)**:
- 由物理拓扑**删掉一组节点**得到(从 10 组预设删除节点集里选,见 §4.2);
- 节点: `comp_capacity≈300`、`cach_capacity≈150`(×0.9~1.1 随机扰动);每个节点随机生成任务
  (`tasks_info_list`: 计算任务 comp / 缓存任务 cach,数量 ~ Poisson(40),资源需求量 uniform;计算 70% / 缓存 30%);
- 链路: `bandwidth≈10 Gbps`、`original_bandwidth≈10`、`weight`(用于 SPF 路由)、`used_bandw=0`;
- `comp_effic ∈ [0.8,1.2]` 节点计算效率。

**多智能体**: `K_num_existing_v_net = 3`(已有网络 k1,k2,k3)+ 1(新网络 g)= **4 个 agent**:
`agent_k1, agent_k2, agent_k3, agent_g`。

### 4.2 样本集生成与缓存(重要! 数据管道枢纽)

- 10 组"删除节点集"硬编码于 `main.py` 构造函数(main.py:~lines 95-104),每组删 11 个节点(剩下 12 节点);
  前 6 组用于生成 k1/k2/k3, 后 4 组用于生成 g。
- **600 个样本**预生成并缓存于:
  `data/GEANT/GEANT_3_<移除节点数>_600_sample_save.pkl`(现有 8/10/12/14/16 几个版本,由 `main_remove_node.py` 等使用不同节点数)。
- 每个样本 = 元组 `(k1_net, k2_net, k3_net, g_net, DG)`(3 个已有网 + 1 个新网 + 物理网副本)。
  存在则直接加载(`assert sample_cnt==600`);不存在则现场生成并 pickle。
- `_set_v_network_attr()`(main.py:~line 1069)/ `_set_g_network_attr()`: 给每个 VN 填容量/任务/邻接,并回写物理网 `allocated_*`。
- 每次 step 会 `sample_idx` 递增轮换样本(0→599→0),训练相当于在 600 个不同"资源竞争场景"上持续采样。

### 4.3 观测空间(每个 agent 240 维, 连续, 范围 [0,1])

每个 k-agent 观测 `obs_k_one_shape = 5*12(节点) + 3*36(链路) + 2*36(拓扑) = 240`:

| 段 | 组成(每个元素均归一化到 [0,1]) |
|---|---|
| 节点 ×12 | `[是否与g重叠(0/1), comp_capacity/物理comp, used_comp/comp_capacity, cach_capacity/物理cach, used_cach/cach_capacity]` |
| 链路 ×36(不足补零) | `[是否与g重叠, bandwidth/物理bw, used_bandw/bandwidth]`,不足 max_num_links(36) 补 0 |
| 拓扑 ×36(补零) | 归一化后的邻接(s,d) 展开 |

g-agent 观测同形状(补零对齐): `[0, 物理已分配comp/物理comp, comp_capacity/物理comp, 物理已分配cach/物理cach, cach_capacity/物理cach, …]`。
全局 state = 4 个 agent 观测拼接(仅供 MAPPO critic; `use_parameter_sharing=True` 共享 actor 参数)。

### 4.4 动作空间(每个 agent 4 维, 连续, [-1,1])

- k-agent: `[节点控制×2, 链路控制×2]`,经 `_get_scale_repeat_action`(main.py:~line 2341)“一对多”重复扩展到重叠节点/链路逐个生效(`control_scale=ceil(len(overlap)/len(action))`)。
  物理效果(节点): `new_cap = 0.2*cap + 0.8*cap*(1-a)`(让出上限 `limit_n_allocate=0.2`);链路同理 `limit_l_allocate=0.2`。
- g-agent: 前 3 维分别控制自身 计算 / 缓存 / 带宽 的调整比例(`limit_*=0.3`),按公式缩容自身需求。
- 全部 `(a+1)/2` 缩放到 [0,1] 后使用。

### 4.5 step 动力学(核心! `step()`, main.py:~line 311)

每步流程:
1. **k1/k2/k3 依次"让出"**: 对各自与 g 重叠节点/链路调用 `_get_k_node_cost` / `_get_k_link_cost` → 计算**让出/迁移代价** `cost_k`,并更新物理网 `allocated_*`;
2. **g 部署**: `_get_g_adjustment`(main.py:~line 1494): 缩容自身节点/链路需求 → 逐个检查物理网剩余资源是否满足(节点 comp/cach 足够、链路剩余带宽足够)→ **全部满足 → `g_done=True`(部署成功)**; 任一不满足 → `g_done=False`(本步收益 0);
3. **收益 & 代价**: `revenue_k = w_n*demand_n + w_l*demand_l`(`w_n=0.01, w_l=10`; demand_n=节点任务总需求, demand_l=TM 流量负载);
   `cost_k = β_n*cost_n + β_e*cost_l`(`β_n=0.01/2, β_e=1/5`); g 的收益 `revenue_g`(仅成功时);
4. **贡献度分配** `_get_contribution_degree`(main.py:~line 1405): 各 agent 让出的资源/g 需求之比 → 逐节点/链路归一化 → `lam_n=0.6`(节点)+`lam_e=0.4`(链路)→ 归一化得 `[c1,c2,c3,c_g]`(物理网剩余资源以 0.1 权重参与);
5. **效益与奖励**(main.py:~line 531-553):

```
V_k  = revenue_k − cost_k + X_k          # X_k = revenue_g * contribution_degree[k]   (分成)
V_k0 = 让出前的旧收益(动身前)
reward_k = contribution_degree[k] * (V_k − V_k0)/V_k0 * 10 * 4      # clip 到 [-5,5]
V_g  = revenue_g − X_k1 − X_k2 − X_k3
reward_g= contribution_degree[3] * (V_g − 0)/20 * 1.5              # g_done 才非零, clip [-5,5]
```

   未成功(`g_done=False`)时 reward 一律 0 → 形成“只有整体部署成功才有回报”的稀疏合作信号。
6. **进入下一样本/下一观测**, 累计记录 `reward_*_list`、`g_done_list`、`dict_train_index`(V / V_k / X_k / cost / revenue 等收敛指标)。

### 4.6 代价细节(论文公式的来源, 便于论文章节描述)

- **节点让出代价 `cost_n`**(`_get_k_node_cost`): 缩容后节点任务溢出 → 依次把任务**迁移到剩余资源最多的 1 阶/2 阶邻居**(按剩余资源排序选能装下且路径带宽足够的),迁移代价由
  `τ_mig + τ_q + τ_pd − τ_ps`(迁移/排队/处理时延)+ `γ·资源量` 组合; 无处可迁则按 **50** 惩罚。 
- **链路让出代价 `cost_l`**(`_get_k_link_cost`): 带宽缩容 → 链路 weight 按 `weight *= old_bw/new_bw` 更新 → 用 dijkstra 重算路由 → 与旧路径对比得到**重路由流量量** → 组合出 收敛代价 `ρ_cvg`(与重路由比例/网络规模/路由器性能相关)+ **负载均衡代价**。
- 流量分配: `_ksp_traffic_distribution`(按 TM 在 k 最短路径上均分)与 `_ecmp_*`(ECMP 递归均分)两种。

### 4.7 评估模式

- `is_eval` 标志; 每 `eval_max_step=512` 写 `eval_marl_env.pkl`; 评估指标主要为 **接受率 acceptance rate = mean(g_done)**、`V`(总效益)、`V_k`、`V_g`(mean over 512)。

---

## 5. 训练流程与超参(main.py `__main__`, ~line 3071)

- `parse_args` → 读 `configs_main.yaml` → 拼路径(`result_root/model_save|env_save|reward_save|logger_save|...`, 均会自动 mkdir)→ `setlogger`。
- `configs_main.yaml` 关键超参: `agent: MAPPO`, `policy: Gaussian_MAAC_Policy`, `representation: Basic_MLP`,
  `fc_hidden_sizes [256,256,128]`, `learning_rate 0.0007`, `gamma 0.99`, `n_epochs 10`, `buffer_size 1024`,
  `use_parameter_sharing True`, `device cpu`, `running_steps 10_000_000`, `max_num_steps 512 000`, `eval_max_step 512`。
- 若 `model_dir` 与 `env_save.pkl` 不存在 → `make_envs(configs)` 用注册进 XuanCe 的 `CustomMultiAgentEnv` 构造向量化环境 → `CustomMAPPO_Agents.train(max_num_steps)` → 存 `.pth` + pickle env; 已存在则加载后 `test_run_episodes`。
- 保存(每次 `timestep==max_num_steps`): `rewards_marl_env.pkl`,`dict_train_index_marl_env.pkl`;评估写 `eval_marl_env.pkl`。
- `CustomMAPPO_Agents`(main.py:~line 2836)继承 `MAPPO_Agents` 并在评估窗内打印 `Acceptance_rate_mean/std`。

**算法对照**: `configs_main_INDRL.yaml`(独立学习器, 各 agent 独立 PPO/critic)、`configs_main_SingleRL.yaml`(单一策略控制全部)、`main.py`(MAPPO 共享策略 + 全局 critic)。

---

## 6. 数据文件与格式(不随本文提供, 说明以便理解/再生成)

- **拓扑**: `data/GEANT/topology_file/topology-anonymised.xml` → `GEANTTopology` 解析成:
  - `GEANT`(文本, 每行 `Link_index\tSource\tDest\tWeight\tCapacity(kbps)\t(node_id,node_id)`)
  - `GEANT_shortest_paths`(每行 `s->d: [多条最短路径]`)
  - 内存 `DG`(networkx DiGraph)、`link_idx_to_sd`、`pair_sd_to_idx`、ECMP 下一跳 等。
- **流量矩阵**: `data/GEANT/2005_04_1st_week|2nd_week/IntraTM-*.xml`(每周 672 个 XML)→ `GEANTTraffic` 合并成文本
  `GEANT_TM1 / GEANT_TM2`(每行 = 23×23=529 个数, 空格分隔, 单位 kbps)→ `traffic_matrices[TM序号, ·, ·]`。
- **样本 pkl**: `data/GEANT/GEANT_3_12_600_sample_save.pkl`(3=K, 12=移除节点数, 600=样本数), 结构见 §4.2。
  (**约几 MB/个, 可选提供; data/ 整体数 GB 主要是那 1300+ 个 XML。**)
- 物理网链路容量在 `main.py` 里被改写为 33 Gbps; 虚拟网 ~10 Gbps; TM 在环境中除以 1e6(kbps→Gbps)。

---

## 7. 实验家族全景(main_*.py 用途归类)

| 类别 | 文件 | 使用的环境/配置 | 用途 |
|---|---|---|---|
| ★ 主方法 | `main.py` | `CustomMultiAgentEnv` · `configs_main.yaml` | MAPPO(共享参数 + 全局 critic) |
| 独立学习 | `main_INDRL.py` `main_INDRL_kn.py` `main_INDRL_rn.py` | `configs_main_INDRL.yaml` | 各 agent 独立策略(对比共享) |
| 单智能体 | `main_SingleRL.py` `_kn` `_rn` | `configs_main_SingleRL.yaml` | 去掉多智能体合作的对比 |
| 可扩展/规模 | `main_ScaleDRL.py`, `*_r1sc.py`(DRL-TE_r1sc, custom_environment_scaledrl_r1sc…) | 对应 `custom_environment_*` | 论文"规模化/R1 场景"实验 |
| 静态基线 | `main_SPF.py` | `utlis/custom_environment_spf.py` | SPF(最短路径)等无需学习的基线 |
| DRL-TE 基线 | `main_DRL-TE.py` `main_DRL_TE_r1sc.py` | `utlis/custom_environment_drlte*.py` | 经典流量工程 DRL(TE)基线 |
| 消融/敏感性 | `main_remove_node.py`(移除节点数 8/10/12/14/16)、`main_k-net-number.py`(K 的个数)、`main_physical_resource.py`(物理资源量)、`main_link_flow_num.py`、`main_link_selection_method.py`(链路选择策略)、`main_coarse-grain.py`(粗粒度粒度)、`main_save.py` | 各自环境 | 各维度敏感性/消融 |
| 共用工具 | `utlis/custom_environment*.py`、`utlis/config.py`、`utlis/general_function.py` | — | 基线环境/工具 |

> `_kn`/`_rn` 后缀大概率分别代表 "K 数(k_net number)" 与 "移除节点数(remove node number)" 变体(文件名即可见); 具体每个变体改动的量, 以实际源码为准, 可让 GPT 对照发送的源文件确认。

---

## 8. 基线代码来源提示

`utlis/config.py`、`utlis/custom_environment*.py` 属于**经典开源的 "CFR-RL / 深度强化学习做流量工程(TE)" 风格框架**(Conv 观测、actor-critic、TE_v2 版本),`replay/finetune/transfer_generator.py` 来自早期 **MAML(模型无关元学习)小样本故障分类**研究工作。这些不属于主方法, 主要用于论文"对比基线/先验工作"章节背景。

---

## 9. 用大模型(LLM)替换某环节的候选切入点(供 GPT 论证)

下面列出本项目中**手工设计/启发式/经验系数**最集中的位置——这些是最自然的 "可被 LLM 增强或替换" 的对象:

1. **贡献度/收益分配(§4.5 的 `contribution_degree`, 类 Shapley)**: 目前是指定权重(0.1/0.6/0.4)+ 归一化的解析式 → 可尝试 "LLM 协商 / 语义化公平分配 / LLM 引导的 Shapley 估计"。
2. **任务迁移调度(§4.6 节点代价中的迁移目标选择)**: 目前是 "剩余资源最多且路径带宽够" 的贪心 → LLM 可做 QoS/时延感知的语义决策。
3. **g 的部署策略(agent_g 的动作语义 / 成败判定)**: 仅 3 个标量 → 可升级为 LLM 给出的结构化部署/缩容方案。
4. **稀疏奖励与手工系数(β、w、lam、0.1、10/4/1.5 等)**: 可让 LLM 做奖励塑形解释或自适应系数。
5. **多智能体之间的冲突仲裁**: k1/k2/k3 同时让出 → 可能出现资源竞争, LLM 作为协调者/中介。
6. **可解释性/论文增量**: LLM 将标量策略输出翻译为人类可读的 "为什么这么分配" 的决策解释 → 提升论文故事性。
7. **场景/任务生成(§4.2 的随机样本与系数)**: LLM 生成更有挑战或更贴近真实业务的虚拟网/流量场景。

> 以上仅作起点。请 GPT 据此为该开发者: 选定 1 个替换点 → 给出技术方案(输入/输出/训练方式: LLM-in-the-loop、LLM 辅助分层、LLM 策略蒸馏等)→ 设计对比实验与贡献点 → 起草论文章节结构 → 指出实现时要改的 `main.py` 具体函数与潜在风险。

---

## 10. 常见坑与注意事项(避免后续开发踩坑)

- `configs_main.yaml` 中 `root` 是**硬编码的 Linux 路径**(`/home/ubuntu/...`),换机器必须改; 同理 `utlis/config.py` 里也有 Windows 硬编码。
- `max_num_steps=512000`、`running_steps=10M` 非常长, 调试建议先把 `sample_cnt`/步数改小。
- 环境状态是**有状态的连续过程**(物理网 `allocated_*` 会累加), 复现依赖 `setup_seed(seed)` 与确定性的 `sample_dict` 顺序。
- `sample_cnt==len(sample_dict)` 有 assert, 若手动增删 pkl 需一致; 每次 `sample_idx` 到 600 会重读 pkl。
- 众多元代码(agent_k1/k2/k3 三段几乎复制)重构空间大, 但不建议论文期大规模重构。
- 训练产物(模型/评估 pkl/日志)写入 `result/`、`logs/`、`figure/`、`marl_models/`, 均可事后查看收敛曲线。

---

## 11. 附: 建议提供给 GPT 的文件清单(详见 `WHAT_TO_SEND_GPT.md`)

| 优先级 | 文件 | 大小估 | 必需? |
|---|---|---|---|
| ★1 | **本文件 `PROJECT_DIGEST.md`** | < 100 KB | 必 |
| ★2 | `main.py` | ~250 KB | 强烈建议(环境+训练全貌) |
| ★3 | `configs_main.yaml` / `configs_main_INDRL.yaml` / `configs_main_SingleRL.yaml` | ~15 KB | 建议 |
| ★4 | `utlis/network_env.py` | ~120 KB | 可选(数据/拓扑管道) |
| ★5 | `requirements.txt` / 关心的 1–2 个 `main_*.py` 变体 | ~100 KB | 可选 |
| — | `data/`(XML/pkl) | **数 GB** | **不提供**(本文件已足够) |
