# 基于 LLM 替换 GEANT 资源分配模块 —— 研究方案

> 目标：在本地 MAPPO 合作博弈 MARL 项目上，用大模型替换某一模块并形成论文。
> 本文件是基于现有 `main.py` 真实代码的可行性分析与实施方案。

---

## 0. 结论先行（TL;DR）

**推荐替换点：`贡献度 / 收益分配`（`main.py::_get_contribution_degree`，line 1405）→ 换成 LLM 引导的语义化 Shapley 分配 / 协商层。**

理由：
1. 它是全项目里"类 Shapley + 手工系数 + 可解释故事"最集中的位置，也正好是你和 IEEE WCL 那篇**差异最明显**的地方。
2. 改动面**最小、最可回滚**：只改 `step()` 里一处调用 + 新增一个 `llm_interface.py`，MAPPO 训练主体完全不动。
3. 该返回值在 `step()` line 545-548 直接作为 k1/k2/k3/g 的奖励权重 → **替换它就同时重塑了整条合作关系**，可做 4 种分发策略的对照实验。
4. 不必训练大模型（免 GPU/免 RLHF 基建，适配 torch1.13 老旧环境）——用 **In-the-Loop（检索/提示式）+ 可选近端 DPO 对齐**，比 IEEE 那篇"双 LLM SFT+DPO"更轻、转移点更聚焦。

> 方案 B（任务迁移的 QoS 语义调度，`_get_k_node_cost` line 1786-2036）和方案 C（g 部署策略 `_get_g_adjustment` line 1494）作为**实验的消融/扩展**，不建议作为主替换点：改的是环境内部动作构造，会牵动收敛性，风险高、可解释性故事弱。

---

## 1. 现状代码定位（证据链）

### 1.1 贡献度=类 Shapley 的解析式分摊
`main.py:1405` `_get_contribution_degree(assigned_resoure, unassigned_resoure)`：

```python
p_n_w_factor = 0.1; p_l_w_factor = 0.1   # line 1427/1441 物理剩余资源折算
lam_n = 0.6; lam_e = 0.4                    # line 1450-1451 节点/链路权重
contribution_degree = lam_n * cd_node + lam_e * cd_link
contribution_degree /= contribution_degree.sum()   # normalize → [k1,k2,k3,g]
```

### 1.2 贡献度 = 奖励权重（杠杆点）
`main.py:514-548`（`step()`）：
```python
contribution_degree = self._get_contribution_degree(...)
X_k1 = revenue_g * contribution_degree[0]       # 分给 k1 的份额
...
reward_k1 = contribution_degree[0]*(V_k1-V_k1_0)/(V_k1_0+EPS)*10*4 if g_done else 0
reward_g   = contribution_degree[3]*(V_g)/20*1.5
```
→ **contribution_degree 以"乘法权重"出现在每个 agent 的收益/奖励里**。LLM 只需输出一个 4 维 ([0,1]^4, sum=1) 向量即可无缝接入。

### 1.3 其余手工系数密度
- `limit_n_allocate = 0.2`（line 1711，k 让出上限）/ `0.3`（line 1514，g）
- 迁移代价神秘常数：放弃任务 `cost_mig_nm = 50`（line 1889）、时延系数 0.5/0.2/1/32/5（line 1902-1926）
- 奖励常数：`0.01/10`、`/20`、`1.5`、`*10*4`、`beta_n=0.01/2, beta_e=1/5`（line 423-425）
- 这些可统一交给"LLM 自适应系数"方案做扩展（见 §6 方案 B）。

---

## 2. 与 IEEE WCL 2026 的差异化（论文立足点）

| 维度 | IEEE WCL 2026（Han） | 本方案（你） |
|---|---|---|
| LLM 角色 | **策略/控制器**：Intent LLM 意图→切片配置；Agent LLM 状态→带宽分配 | **分配/谈判层**：LLM 只做"收益如何在 4 个网络间分"；底层让出策略仍由 MAPPO 学 |
| 问题类型 | 单控制器网络切片 + 资源分配（Bellman/MDP） | **合作博弈式多智能体**（多租户 VN 嵌入）+ 类 Shapley 分成 |
| 训练 | SFT 预训练 + 环境奖励 DPO（需训练基建） | **In-the-Loop 提示式（零训练）为主**，可选 DPO 对齐；适配 torch1.13 |
| 优化/评估目标 | satisfaction ratio / total reward | ①总效用 ②**公平性**（Gini/Jain/无嫉妒性 envy-free）③**稳定性/核（core）可行性** ④可解释（LLM 返回分流理由） |
| 场景 | Slicesim/Open5GS，eMBB/URLLC/voice/mMTC | GEANT 23 节点，k1/k2/k3 重叠让出 + g 部署 |

**一句话差异化**：*“IEEE 用 LLM 学策略；我们用 LLM 做多租户资源博弈的收益分配/协商（语义化 Shapley），把 MAPPO 的让出策略和 LLM 的分配层解耦，并首次在公平性/稳定性下评价 LLM 分配。”*

---

## 3. 推荐架构：LLM 语义化贡献度分配（方案 A）

### 3.1 三级落地方案（从轻到重，论文铺量）

**A1（MVP，必做）LLM-In-the-Loop 分流**
- 新增 `llm_interface.py`：封装接口 + 状态→语义摘要 + JSON 解析 + 校验/降级。
- 在 `_get_contribution_degree` 处二选一：
  - 训练期：仍用解析式（快、可复现）。
  - 测试/eval 期：把"k1/k2/k3 各让出多少节点计算/缓存/带宽、g 占用多少"→ 自然语言摘要 → LLM 输出 `[w_k1,w_k2,w_k3,w_g]` → 归一化后**替换** 解析式贡献度。
- 校验：`sum≈1`、`0≤w≤1`；解析失败 → 回退解析式（保证训练不崩）。
- 论文价值：直接对比"解析式 Shapley vs LLM 语义 Shapley"的分流质量 + 可解释。

**A2（中，增强）协商式公平分配 + 无嫉妒性度量**
- prompt 里给每个网络"语义身份 + 让出代价 + QoS 诉求"，让 LLM 做**多轮/单轮谈判**（可模拟 4 个子 LLM 子代理协商出一致分流）。
- 评价是否 **envy-free**：k 对 k' 的份额占比是否与其让出贡献匹配；算 **Gini/Jain**。
- 用"LLM 分流"当作**训练期奖励**跑一段 MAPPO，对比收敛（见 §5 实验 3）。

**A3（强，论文加分）近端对齐 / 提炼**
- 若能跑训练：收集 `(语义摘要 → 解析式最优分流)` 作为**偏好对**，用轻量 DPO/LoRA 对齐一个小 LLM，让输出的分配同时满足"公平 + 接近解析最优"。
- 或：把 LLM 分流蒸馏成一个小 MLP head，替换贡献度，训练时不再阻塞。

### 3.2 需要改动/新增的文件（最小集）
| 文件 | 改动 |
|---|---|
| `llm_interface.py`（新增） | 状态→摘要、调用、JSON schema、校验、cache；**用 `openai` SDK 连研究室内网网关**（OpenAI 兼容） |
| `main.py` | ①`step()` line 517-519 处按开关用 LLM 分流替换 `_get_contribution_degree`；②新增 `self.use_llm_contrib` 配置 |
| `llm_config.yaml`（新增） | **base_url=`https://llm.example.invalid/v1`**、分流主模型、解释模型、`api_key`（经环境变量注入，**不进 git**）、temperature(0)、开关、回退策略 |

**已确认环境（研究室本地化模型网关，V1.7）**：
| 用途 | 模型 | 说明 |
|---|---|---|
| 分流主模型（循环内） | `qwen3-next-80b-a3b-instruct` | 256k 上下文、非推理 → 输出快、确定性好 |
| 代码/分析（dev 用） | `qwen3-coder-next` | 256k |
| 可解释/推理 | `deepseek-v4-flash-0731` | 1M 上下文、推理型（本会话自身即为该模型） |
| Embedding（可选 RAG） | `qwen3-embedding-8b` | 4096 维、32k 上下文 |

> **额度**：网关全部模型**额度不限** → 训练/实验不再被 API 成本约束；数据可规模化（A3 偏好对、长时 eval 均可放开）。唯一现实约束是**每步调用时延**。

### 3.3 关键设计/坑
- **阻塞与速度**：step 同步调用 → 训练慢。方案：训练用解析式、eval 用 LLM（A1 天然规避）；A3 蒸馏后不阻塞。（额度不限≠时延不限。）
- **可复现**：temperature=0 / seed；循环内用**非推理型** `qwen3-next`（推理型 `deepseek-flash` 会额外产出 reasoning token，时延更高，仅用于解释/协商）。
- **合法性**：LLM 可能输出非法比例 → 必须 `softmax`/归一化 + 数值兜底。
- **随机样本**：600 sample 轮换，语义摘要必须**确定性**构造（不要喂采样方差）。
- **鉴权与安全**：`api_key`（sk-XXX）只经环境变量注入 `llm_interface.py`，`llm_config.yaml` 占位；`api_key`/令牌不提交 git。

---

## 3.4 硬件评估与"是否微调"决策

**本机实测**：无独立 GPU（仅 Intel UHD 630 核显）；i7-9700 8C8T；32GB RAM。→ **本机不能做本地 LLM 部署/微调**（无 CUDA，CPU 上 7B 仅 5-8 tok/s）。**不要在本地装 Ollama。**

**决策结论**：
- **主线 A1/A2 完全不需要微调**：纯提示工程 + structured output 即可达到可发表水平；"免训练 In-the-Loop LLM 接入"本身就是与 IEEE WCL 2026（需 SFT+DPO 训练）的差异化卖点，也适配本机无 GPU。
- **A3"轻量对齐/蒸馏"的唯一合理地点是部门 GPU 服务器**（`192.168.131.35` 网关即 vLLM 起，说明部门有 GPU 机）——若可用：用网关生成偏好对数据集 → 在该机 LoRA（勿在本机）。
- **本机 ultra-lite 蒸馏替代**（强烈建议作为"训练"卖点）：用 `qwen3-embedding-8b` 把语义摘要编码为 4096 维向量 → `main.py` 的 torch 环境里训一个**小 MLP 头**预测 4 维分流 → 替换贡献度。本机 CPU 即可训，且与现有 MARL 代码天然契合。该方案即"LLM 蒸馏/规约"，可作为实验 5 的落地而非微调大模型。

---

## 4. 实验规划（对照表）

| # | 实验 | 替换对象 | 独立变量 | 指标 |
|---|---|---|---|---|
| 1 | Baseline | — | 解析式贡献度 (0.1/0.6/0.4) | V, reward, g_done |
| 2 | Equal | 解析式 | 平均分摊 (1/4,1/4,1/4,1/4) | 同上 + 公平性 |
| 3 | **LLM-In-the-Loop** | 解析式 | A1 分流 | 总效用 + Gini/Jain + g_done |
| 4 | **LLM 协商** | 解析式 | A2 多子代理协商 | + envy-free + 协商轮数/时延 |
| 5 | **LLM+DPO/LoRA**（可选） | 解析式 | A3 对齐小模型 | 分流质量 + 公平 + 时延 |
| 6 | LLM 自适应系数 | 各手工系数 | 用 LLM 动态调 β/limit/w | 收敛速度 + 稳定性（方案 B） |
| 7 | LLM QoS 迁移（扩展） | `_get_k_node_cost` | LLM 选迁移目的节点 | 迁移代价 + 时延（方案 C/L 片段） |

**统一指标**：①g 部署成功率 `g_done`；②总效用 V=V_k1+V_k2+V_k3+V_g（`dict_train_index['V']`）；③公平性 Gini/Jain/无嫉妒性；④稳定性（核/core 是否非空、是否分配可知的 enough）；⑤LLM 时延/步；⑥可解释示例（LLM 的分流理由）。

---

## 5. 风险与对策

| 风险 | 等级 | 对策 |
|---|---|---|
| LLM 阻塞训练 / 速度 | 高 | 训练解析式 / eval 用 LLM（A1）；或蒸馏 head（A3） |
| 非确定性 → 论文图表波动 | 中 | temp=0、固定 seed、多次运行取均值±std |
| LLM 非法/不稳定输出 | 中 | JSON schema + softmax 归一化 + 解析失败回退解析式 |
| 公平性与总效用此消彼长 → "LLM 可能更差" | 中 | 把它写成**可分析的 trade-off**（公平换效用的帕累托曲线），本身就是贡献 |
| 微调需 GPU/基建与 torch1.13 冲突 | 中 | 默认不做训练（A1/A2）；A3 仅在可用环境做，不阻塞主线 |

---

## 6. 备选方案（如你想换主线）

- **方案 B：LLM 动态系数 / 奖励塑形**。改动最小的"软替换"：LLM 每步输出 `{limit_n_allocate, lam_n, lam_e, beta_e, ...}` 的自适应值。故事弱，适合做消融。
- **方案 C：LLM QoS 感知迁移**（`_get_k_node_cost` line 1828-1885 现为"剩余资源最多+带宽够"贪心）→ LLM 综合时延/利用率选迁移目的节点。落地直观但改环境内部、要重训，风险高。

**推荐主线：A1→A2→（可选）A3；B/C 全部降级为 ablation 填充实验表格。**

---

## 7. 下一步执行的 10 步清单

1. `git branch llm-contrib`（或 worktree），冻结 baseline；`git stash` 现有 .gitignore 改动。
2. 交付本方案的 `docs/llm_replacement_plan.md` 定稿。
3. 写 `llm_interface.py`（`openai` SDK 接研究室内网网关 `https://llm.example.invalid/v1`；循环内 `qwen3-next-80b-a3b-instruct`、解释用 `deepseek-v4-flash-0731`）。
4. 构造 5-10 个**代码内真实样本**的语义摘要，手工验证 LLM 输出分流是否合理。
5. 在 `main.py` `step()` 加 `use_llm_contrib` 开关 + 归一化/回退（不破坏训练主链）。
6. 跑实验 1(Baseline/解析式) 拿到可复现 baseline 曲线。
7. 跑实验 2/3（LLM 分流，eval），记录 V、g_done。
8. 跑实验 4（协商）+ 公平性/稳定性指标脚本。
9. （可选）A3 蒸馏：`qwen3-embedding-8b` 摘要向量 + 本机 CPU 训小 MLP 头替换贡献度；或部门 GPU 机上 LoRA（若可用）；再做 LLM 自适应系数 ablation。
10. 起草论文骨架（§7 提纲）。

---

## 8. 论文故事骨架（初版）

**题目（候选）**：《LLM 引导的多租户虚拟网络嵌入收益分配：语义化 Shapley 与公平性评估》
**结构**：
1. Intro：GAN 上多 VN 协作、稀疏奖励、类 Shapley 分摊的手工系数局限。
2. Related：DRL/MARL 切片 vs LLM×切片（点名 IEEE WCL 2026）→ 我们 differ：LLM 在分配/博弈侧。
3. 系统模型：4-agent 合作博弈，贡献度=Shapley-like，LLM 分流。
4. 方法：语义 state 编码 → LLM 分流/协商 → 校验/回退 → MAPPO 底层让出策略。
5. 实验：效用 / 公平性(Gini、Jain、envy-free) / 稳定性(core) / 可解释 / 时延。
6. 结论 + ablatation（B/C）。

---
*本文件基于 `main.py` 现状（贡献度 line1405、step 奖励 line514-553、迁移 line1786-2036）撰写。*

---

## 9. 第一篇小论文方案（主线 A1，不依赖微调）

**定位**：以"语义化收益分配 + 公平性评估"为**唯一**卖点的小论文（中文核心 / EI 会议体量）。
- 不需要微调：LLM 走内网 API 做分流；MAPPO 训练 + g 部署都有项目现有代码支撑。
- 复用现有对照：`main_INDRL.py` / `main_SingleRL.py` / `main_SPF.py` 可作为训练环境/基线的佐证，但**论文 1 只固定用 MAPPO，仅变分配策略**这一根轴，故事最干净。

**完整故事链**：
1. **问题**：多个 VN 在 GEANT 重叠节点/链路上"让出"资源部署新网 g；收益分配是手工加权类 Shapley（0.1/0.6/0.4），对"让出代价 / QoS 损伤"无语义理解，公平性与可解释性差。
2. **方案**：LLM-In-the-Loop 语义分流——把各网络让出量 + g 占用构造成**确定性语义摘要** → LLM 输出 4 维贡献度 → 归一化/校验/回退 → 作为收益与奖励权重（替换 `_get_contribution_degree`）。
3. **评估**：g 部署成功率 `g_done`、总效用 V、公平性（Gini/Jain/envy-free）、可解释示例、每步时延。
4. **卖点**：LLM 让"分配+理由"对齐人类直觉，公平性比解析式更平滑，且全程免训练（相对 IEEE WCL2026 需 SFT+DPO 的负担）。

**实验矩阵（同一 MAPPO 设置，只变分配策略）**：
| 策略 | 做法 | 预期卖点 |
|---|---|---|
| 解析式（现状） | `_get_contribution_degree` | baseline |
| 平均分摊 | 常数 1/4 | 反例：无信息分配 → 展示不公平/低效用 |
| **LLM 分流 (A1)** | `llm_interface.py` | 主结果：公平 + 可解释 |
| **LLM 协商 (A2)** | 多子代理 | 加分：envy-free 情节 |

**必备前置**：跑通 §7 步骤 4 的 prompt 用例 → 同 seed 先拿解析式 baseline 曲线 → 再补 3 条策略曲线。
**投递备选**：中文核心（电子学报 / 通信学报 / 计算机应用）、EI 会议均可；体量以 6-8 页实验论文为佳。

---

## 10. 微调到底有没有办法？（三条路 + 概念提醒）

| 路径 | 可行性 | 说明 |
|---|---|---|
| **本机** | ❌ | 已实测无独显（仅 UHD630 核显、i7-9700、32G），CPU 训 7B 不现实 |
| **部门网关服务器** | ❓待确认 | `192.168.131.35:30001` 是 vLLM **推理**网关；要 LoRA 需另问管理员是否有 GPU 训练机/账号/磁盘 |
| **租卡（AutoDL 等）** | ✅ 最实际 | 4090/A10 24G 约 1-2 元/时；`llama-factory` / `unsloth` 一键 QLoRA 7B，几千-几万条即可训 |

**但论文 1 不需要微调**，且有一个概念陷阱要先想清楚：
- 若监督信号用"解析式最优分流"，等于**用你要替换的启发式去教 LLM** → 自我循环，没意义。
- 更有意义的信号是"环境总效用/牺牲公平性"那类（IEEE 用 DPO 的做法），但要把**训练闭环接进实验脚本**，工作量超出论文 1 体量。

**微调真正值得做的场景（留给第 2 篇/期刊）**：把分流能力**蒸馏**到小模型或本机 MLP 头，使运行时**离线、低时延、不依赖内网 API**——这是"系统效率/轻量化"故事，不是论文 1 的卖点。
