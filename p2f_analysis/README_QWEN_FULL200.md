# Qwen 完整策略扩大实验

使用既有 `p2b_val_seed5601` 全部200验证场景，两组各800次角色调用，共1600次。原提示词与上一轮冻结链路校准提示相比，只有已有切片 system 追加段不同；g消息相同，但四角色都重新推理，完整应用新动作，不固定旧节点或g。

固定 Qwen 模型、Standard-1000 P2-A 检索、角色自身训练历史IR、3个示例、temperature0.1、max_tokens4096、环境种子1。请求串行，无自动重试，无fallback。不得在运行过程中修改锁定来源或候选Prompt。

API调用前必须重建全部800份角色消息及示例ID，并在原环境核验400个历史Qwen/MAPPO控制。checkpoint无需加载。原环境周期性保存重定向到新目录。原数据/奖励/指标/迁移/重路由实现保持不变。

实际已启动目录：`p2f_analysis/qwen_full200/run_20261104T141915_396460Z/`。完成与否以summary.json和审计结果为准。不要通过重复启动试验查看进度。

## 命令

离线契约测试，无API：

```powershell
& 'python' -B -m unittest discover -s .\p2f_analysis -p test_qwen_full200.py -v
```

单独完整预检，无API；创建新预检目录，读取验证样本和训练经验：

```powershell
.\p2f_analysis\run_qwen_full200.ps1 -Mode check
```

实际推理命令，每次会新建目录并产生1600次Qwen请求，失败即停，不自动恢复：

```powershell
.\p2f_analysis\run_qwen_full200.ps1 -Mode run
```

完成后的离线审计，首次写入audit_and_paired.json、comparison.csv、paired_scene_metrics.csv和RESULTS.md，不调用API：

```powershell
& 'python' -B .\p2f_analysis\audit_qwen_full200.py --run-dir .\p2f_analysis\qwen_full200\run_20261104T141915_396460Z
```

## 解释范围

主要指标为合作有效率和V，次要指标为接纳率、V_g和成本。保留全部失败与IR违规。对同场景做配对比较，区间为探索性且不涵盖同场景模型多次采样的完整不确定性。两组串行执行也可能有服务时段波动。

全200场景均已进入开发分析；前40用于上一轮小实验，后160也曾做离线诊断，不能称独立测试。冻结后新独立评估需另设协议，不使用P2-C选择参数。按用户内网无限额度说明新增API费估计为$0；若usage缺失则明确记录token未知。
