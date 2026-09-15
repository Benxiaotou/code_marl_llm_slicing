# 交给 GPT 的文件清单 · 使用说明

> 目标: 让 GPT **不需要**那几 GB 的 `data/`,即可完整理解项目并指导后续 "LLM 替换某环节" 的论文开发。
> 全部建议内容合起来 **约 1–2 MB**,远在你的 "几十 MB 以内" 限制之下。

---

## 一、最终要发送给 GPT 的内容(总共 ~1–2 MB)

| 顺序 | 文件 | 大约大小 | 作用 | 必要度 |
|---|---|---|---|---|
| 1 | `gpt_deliverable/PROJECT_DIGEST.md` | < 100 KB | 项目全解说明书(问题建模/环境/奖励/代码/实验家族) | ★ 必备 |
| 2 | `main.py` | ~250–350 KB | 环境 `CustomMultiAgentEnv` + 训练循环(全文) | ★ 强烈建议 |
| 3 | `configs_main.yaml` | ~5 KB | MAPPO 超参 | 建议 |
| 4 | `configs_main_INDRL.yaml` | ~4 KB | 独立RL 对比配置 | 建议(提到对比时有用) |
| 5 | `configs_main_SingleRL.yaml` | ~4 KB | 单智能体RL 对比配置 | 建议 |
| 6 | `utlis/network_env.py` | ~120 KB | 拓扑/流量矩阵解析、删点生成子拓扑 | 可选(想深挖数据管道时给) |
| 7 | `utlis/general_function.py` | ~7 KB | 日志/种子 | 可选 |
| 8 | `requirements.txt` | ~10 KB | 依赖 | 可选 |
| 9 | 你**最关心的 1–2 个变体**, 如 `main_SingleRL.py` 或 `main_remove_node.py` | ~100 KB 每个 | 让 GPT 看到你要对比/改造的具体实验 | 可选 |
| — | `data/`(XML / TM / sample pkl) | **几 GB** | — | ❌ **不要发**(说明书 §4.2/§6 已解释格式, 能自动再生成; 若 GPT 坚持要看某个 pkl 结构, 可用下面第 §三 的"pkl 元信息脚本"给它打印结构, 而不是发文件) |

> 如果嫌文件多, **最小可用集 = 只发第 1 项(PROJECT_DIGEST.md)**, 说明书已做到自包含; 加上第 2 项 `main.py` 后 GPT 就能引用具体行号给你精确的改代码建议。

---

## 二、一条命令把上述原文件打包成一两个小文件(在你机器上执行)

路径含中文/空格需注意, 用 PowerShell 在项目根目录执行:

```powershell
Compress-Archive -Path "gpt_deliverable\PROJECT_DIGEST.md","main.py","configs_main.yaml","configs_main_INDRL.yaml","configs_main_SingleRL.yaml","utlis\network_env.py","utlis\general_function.py","requirements.txt" -DestinationPath "gpt_deliverable\for_gpt.zip"
```

执行后把 `gpt_deliverable\for_gpt.zip`(约 1 MB)发给 GPT 即可(再手动补发 `PROJECT_DIGEST.md` 作为第一个消息也可以)。

---

## 三、(可选) 让 GPT 了解 pkl 结构的小技巧(不用发 pkl 文件)

如果你的 GPT 需要知道样本 pkl 长什么样, 在你自己环境里跑一下把输出贴给它:

```python
import pickle
with open('data/GEANT/GEANT_3_12_600_sample_save.pkl','rb') as f:
    d = pickle.load(f)
k1,k2,k3,g,DG = d[0]
print('样本数', len(d))
print('k1 节点', k1.num_nodes, '链路', k1.num_links)
print('物理网节点', DG.number_of_nodes(), '链路', DG.number_of_edges())
print('节点属性示例', dict(list(k1.DG.nodes(data=True))[0]))
print('链路属性示例', dict(list(k1.DG.edges(data=True))[0]))
```

---

## 四、建议随文件一起发给 GPT 的提示词(可直接复制)

> 我已附上一份完整的项目说明书(`PROJECT_DIGEST.md`)+ 核心源码(`main.py` + 3 个 yaml + `network_env.py`)。
> 这是一个 "物理网络上 3 个已有虚拟业务网络(k1/k2/k3)给新虚拟网络(g)让出计算/缓存/带宽资源, 用 MAPPO 多智能体强化学习做资源让渡与收益分成(贡献度)" 的论文实验项目。
> 我的目标是: 以该项目为基础, 用大模型(LLM)替换其中某一部分工作, 形成一篇新论文。
> 请你:
> 1) 先用 300 字确认你理解的问题、智能体、奖励公式与实验结构(尤其 `main.py` 里的贡献度/代价/奖励与各 `main_*.py` 变体);
> 2) 基于说明书 §9 的候选接入点, 给出 2–3 个最有论文价值且实现可行的 "LLM 替换/增强" 方案(说明它替换了哪个函数、输入输出、训练范式、期望贡献点);
> 3) 对最有把握的方案给出分阶段实施计划(改哪个文件哪个函数、如何小规模验证、如何与现有基线 SingleRL/INDRL/SPF/DRL-TE 对比、论文章节大纲);
> 4) 指出 3 个最大的风险和你建议先做的最小实验。

---

## 五、小提示

- **先别一上来就让 GPT 设计最终方案**。建议分两轮: 第一轮让它**复述机制 + 提问**, 第二轮给方案——防止它误读奖励/代价公式后带偏方向。
- 说明书第 §7 标注了部分变体用途是**按文件名与导入推断的**; 如果你的研究点就在某个变体上, 把对应 `main_*.py` 也发给它核对。
- 论文初稿阶段, 这份说明书即可作为 "相关工作和系统模型" 章节的素材。
