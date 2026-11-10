# Experience-enhanced LLM decisions for network slicing

本项目研究多智能体网络切片资源分配，包括原始 MAPPO 基线、角色局部经验检索、冻结 Qwen 决策，以及受限的节点和链路动作校正。

四个角色各自读取原始 240 维局部观测。当前主线没有在线角色通信、多轮协商或模型微调。保留原环境的奖励、计账效益、迁移和重路由定义。

## 当前发布范围

此版本补齐 P2C 八方法比较、固定局部校正、十二版本归因、新场景确认、补充指标和 GPT/Qwen 开发适配代码。P0–P2B 历史源码继续保留。尚未冻结的 P3 自主实验、原始数据、权重、逐场模型请求/回复和机器配置不在本发布中。

| 入口 | 用途 |
|---|---|
| `main.py`、`configs_main.yaml`、`utlis/` | 原环境与 MARL 基线；main 包含已记录的 NumPy 路径数组相等性兼容修复 |
| `p1a/`、`p1b/`、`p1c/`、`p1d/`、`p2a/` | 局部提示、冻结教师、经验检索、历史 IR 与风险重排 |
| `p2c_qwen8_queue.py`、`README_P2C_Qwen8_中文.md` | 三种子八方法比较 |
| `p2i_node_boundary/`、`p2j_validation/`、`p2l_cached_test/` | 受限节点校正、新验证与既有场景配对比较 |
| `p2m_paper/README.md`、`p2m_paper/PAPER_SUPPLEMENT_REPORT.md` | MAPPO/kNN/Qwen × raw/link/node/both 的完整设计与解释 |
| `p2n_metric_audit/`、`p2o_alternative_metrics/`、`p2p_business_satisfaction/` | 代价、尾部效益与模型内需求满足率 |
| `p2e_prompt/`、`p3_gpt_heterogeneous_dev/` | GPT 提示与模型分工的开发阶段扩展代码 |
| `research_artifacts/` | 精选汇总、协议及来源哈希；不包含逐场原始记录 |

`p2f_analysis/`、`p2g_autonomous/jobs/`、`p2h_v_decomposition/` 和 `p2k_formal/run_formal.py` 提供后续主线导入的辅助实现。不要只复制最终 runner。

## 已保存的主线确认结果

预先锁定的新种子 2026110511/2026110512/2026110513，各 512 场，共 1536 场；所有十二个版本在相同场景上比较。下表直接摘自发布的 `research_artifacts/paper_fresh/summary.json`。

| 方案 | 平均 V | 接纳率 | 平均调整代价 |
|---|---:|---:|---:|
| MAPPO_raw | 341.4238 | 82.10% | 34.6698 |
| Qwen_raw | 339.4260 | 88.87% | 35.0520 |
| Qwen_both | 344.1780 | 94.01% | 35.1835 |
| MAPPO_both | 343.7879 | 88.41% | 34.6732 |
| MAPPO_node | 344.8885 | 88.41% | 36.2727 |

固定 Qwen 组合相对原始 MAPPO 的平均 V 增加约 2.7543，三项预声明比较的 Bonferroni 区间约为 [0.8189, 4.6993]。对同样校正后的 MAPPO 未建立 V 优势；MAPPO 仅节点版的 V 均值更高。应保留全部十二版本，不能把全部组合增益归因于 LLM。

V 是原环境的计账效益，不是实测吞吐量。同一 GEANT 生成分布下的确认不证明跨拓扑泛化。旧 1536 场此前已查看，后续用途是归因分析；本批新确认数据也不能再次用于调参后称作未用过的最终测试。48 场需求遥测试测仅是模型内任务/资源保留统计，不是 SLA 实测。

## 安装和离线验证

历史运行环境记录为 Python 3.8.20、NumPy 1.22.4、PyTorch 1.13.0+cpu、NetworkX 3.1。`requirements.txt` 是历史完整环境快照的便携文本整理，不是经过新环境全量安装验证的锁文件；tmgen 本地来源及未完整固定的依赖见 [复现说明](REPRODUCIBILITY.md)。

```powershell
# 在已安装 Python 3.8 的隔离环境中
python -m pip install -r requirements-offline.txt
python -B tools/run_offline_checks.py
```

离线检查清除模型凭据，并限制外部网络连接；部分适配器测试使用本机 HTTP mock。它们不运行真实环境实验或付费模型调用。验证记录见 [VALIDATION.md](VALIDATION.md)。

## 模型配置

使用环境变量配置，`.env.example` 仅作格式示例，程序不自动加载它。示例域名不能用于真实调用。配置模型服务后，运行推理可能产生费用，先审查场景数、请求数和预算。

```powershell
$env:LLM_BASE_URL = 'https://your-service.example/v1'
$env:LLM_API_KEY = '<your-token>'
$env:P1A_MODEL = 'qwen3-next-80b-a3b-instruct'
# GPT 扩展另设 GPT61_BASE_URL、GPT61_API_KEY；不要写入源码。
```

若用历史 PowerShell 启动器，部分程序仍从 Windows User 环境读取配置；也可按各脚本说明直接调用 Python。PowerShell 的 Python 可由 `MARL_PYTHON` 指定，默认使用 PATH 中的 `python`。

## 数据与完整复现的条件

此仓库提供源码和精选统计。完整复现还需要合法获取的 GEANT 输入、匹配的 tmgen 源码、冻结 MAPPO 检查点、TRAIN 经验库及各数据划分 manifest。历史 runner 会验证精确路径与源文件哈希；整理后的公开副本不自动满足旧锁，不能跳过这些检查。公开清理与原始文件的关系、精选文件恢复位置见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。

新增实验应建立独立输出目录和新的预声明协议，不覆盖历史结果。旧文档中的执行命令是阶段性操作记录，不代表当前机器已具备输入，也不是付费运行授权。

第三方 XuanCe 镜像保留原有 [LICENSE](docs/xuance-master/LICENSE.txt)。本次不新增第三方代码镜像或赋予未核实的数据、权重再分发许可。
