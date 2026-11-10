# P0-v5：原论文指标对齐与公平评估（中文使用说明）

适用项目：`path/to/code_marl_llm_slicing`，Windows PowerShell，`(marl)` 环境。
不重新训练 MAPPO，不修改原 `main.py`、奖励函数或 GEANT 数据。

## 一、为什么现在做 P0-v5

裴杏龙博士论文第 5 章图 5.7 的比较指标包括：

1. 新切片接纳率；
2. 网络总效用；
3. 新切片 g 效用；
4. 已有切片 k 效用。

论文给出 CoMarl 接纳率 `0.8398`（83.98%）、总效用 `341.22`、新切片效用 `25.18` 等。
原 `main.py` 的评估函数使用最近 `512` 个时间步求平均，原 `configs_main.yaml` 设 `eval_max_step: 512`。

**注意**：下面在当前 600 样本文件中取前 512 个，仅对齐样本数和本地环境；
不能未经确认就声称是论文图 5.7 的完全相同评估样本。
论文的 ERA、SingleRL、INDRL 不等同于我们的 LP-only / Fixed+Guard。

## 二、包内文件

- `p0_paper_eval.py`：只读 JSONL，逐场景对齐、输出论文指标、个体理性、配对差值和审计报告。仅标准库依赖。
- `p0_guard_runner_v5.py`：P0-v4 独立运行器的增强版本，**不覆盖**原 `p0_guard_runner.py`：
  - `--fixed-anchor` 真正调用固定比例（不调用 Qwen、不读历史响应）；
  - `--count 512` 直接选择前 512 个样本；
  - `--on-infeasible record` 遇到 Guard 不可行不丢弃失败样本，继续下一场景；
  - Qwen 接口异常也可按失败计入，不会冒充成功动作；
  - 未执行场景的 `V` / `cost` 记录为 **不可得**，绝不虚构。
- `p0_v5_selftest.py`：13 项纯离线回归测试。
- `P0_V5_README_中文.md`：本说明。

## 三、安装（无需覆盖其他文件）

将压缩包内 **四个文件**解压到项目根目录：

```powershell
cd path/to/code_marl_llm_slicing
python p0_v5_selftest.py
```

应该看到 `Ran 13 tests ... OK`。
`wandb`、`torch` 等第三方包的弃用提示不是 P0-v5 测试失败。

## 四、先用已保存的五个场景，立刻生成统一评价表（0 API 调用）

```powershell
python p0_paper_eval.py `
  --run "MAPPO=.\p0_runs\mappo_mini_20261012_163149" `
  --run "QwenRawV1=.\p0_runs\llm_mini_20261013_165729" `
  --run "FixedGuard=.\p0_runs\fixed_anchor_v1_mini5" `
  --run "LPonly=.\p0_runs\lp_only_mini5" `
  --run "QwenGuard=.\p0_runs\qwen_v4_mini5_first" `
  --expected-indices "0,1,2,299,599" `
  --reference MAPPO `
  --output-dir .\p0_runs\p0_v5_audit_mini5
```

说明：这里的 `FixedGuard` 是历史 v1 固定数值日志的**离线回放**，不是在线 Qwen。
虽然旧日志内部标注为 `recorded_qwen_guard`，对照实验必须按固定参考点标记。
下一节 `--fixed-anchor` 则不再依赖历史日志。

输出：

- `report.md`：核心指标和方法学警告；
- `metrics.json`：详细汇总，含每个已有切片的个体理性满足率；
- `per_scene.csv`：原始场景、原有和新切片效用、成本、补偿、接纳和个体理性；
- `paired_vs_MAPPO.csv`：按完全相同的原始场景配对计算 V / 成本差。

如果方法之间不是同一组原始场景，会**报错停止**。

## 五、主要指标的定义（重要）

- `acceptance_rate = sum(g_done)/N`：与原环境接纳定义一致；
- `V_mean_all = sum(V_s)/N`：前提是每个样本都经原环境产生了 V；
- `V_g_mean, V_k1_mean, V_k2_mean, V_k3_mean`：完整样本上的均值；
- `IR_k = (V_k >= V_k_0)`：已有切片个体理性诊断；
- `IR_g = (V_g >= 0)`；
- `cooperative_valid = g_done AND IR_k1 AND IR_k2 AND IR_k3 AND IR_g`；
- `cooperative_valid_rate = sum(cooperative_valid)/N`。

原代码的 `g_done` **并不把上面的个体理性条件全部作为硬接纳条件**。
因此 `cooperative_valid_rate` 是**新增的科研诊断指标**，
不可直接冒充原论文图 5.7 的接纳率。

程序还校验：

- `V = V_k1 + V_k2 + V_k3 + V_g`
- `V_ki = revenue_ki - cost_ki + X_ki`
- `V_g = revenue_g - (X_k1+X_k2+X_k3)`

**失败样本必须计入 N。**
如果 Guard 拒绝提交且未实际执行 `env.step()`，`V/cost` 不存在：
在全样本汇总中记作 `N/A`，只单独报告已评分样本的描述性均值，
**绝不把只对成功场景的效用均值与 MAPPO 完整样本平均直接比较**。

## 六、下一阶段：512 个匹配场景，先跑零 API 基线

请确认存在项目原始文件：

`data\GEANT\GEANT_3_12_600_sample_save.pkl`

如果没有它，先不要尝试 `--dataset full`。它没有包含在之前的小样本压缩包里。

**第一步：MAPPO 512 场景**

```powershell
python p0_runner.py --dataset full --policy mappo --count 512 `
  --output-dir .\p0_runs\mappo_paper512
```

**第二步：固定参考点 + Guard 512 场景，真正固定数值，不调用 API**

```powershell
python p0_guard_runner_v5.py --dataset full --count 512 --fixed-anchor `
  --on-infeasible record --output-dir .\p0_runs\fixed_guard_paper512
```

**第三步：LP-only 512 场景，不调用 API**

```powershell
python p0_guard_runner_v5.py --dataset full --count 512 --solver-only `
  --on-infeasible record --output-dir .\p0_runs\lp_only_paper512
```

**第四步：先审计三条基线，不必急着跑 512 次 Qwen**

```powershell
python p0_paper_eval.py `
  --run "MAPPO=.\p0_runs\mappo_paper512" `
  --run "FixedGuard=.\p0_runs\fixed_guard_paper512" `
  --run "LPonly=.\p0_runs\lp_only_paper512" `
  --expected-range 0:512 --reference MAPPO `
  --output-dir .\p0_runs\p0_v5_audit_512_phase1
```

请先发这个三方法的 `report.md` / 终端汇总。我们重点确认完整样本内的 Guard
可行率、个体理性保障、与 MAPPO 的真实差异，以及是否存在未评分的拒绝场景。

## 七、Qwen 512 场景（基线审计无重大问题后再跑）

先做前 20 场景的独立小规模冒烟测试：

```powershell
python p0_guard_runner_v5.py --dataset full --count 20 `
  --model qwen3-next-80b-a3b-instruct --prompt-version p0_v3b `
  --api-timeout 20 --max-tokens 2048 --on-infeasible record `
  --output-dir .\p0_runs\qwen_guard_smoke20
```

再做真正的 512 场景测试：

```powershell
python p0_guard_runner_v5.py --dataset full --count 512 `
  --model qwen3-next-80b-a3b-instruct --prompt-version p0_v3b `
  --api-timeout 20 --max-tokens 2048 --on-infeasible record `
  --output-dir .\p0_runs\qwen_guard_paper512
```

API 失败、Guard 无可行解的情况都会计入失败总数。
**本地 API 读取超时并不等于严格的端到端总超时**，若服务卡住，
可手动中断；不完整运行必须补齐后才能形成 512 场景比较。

之后与其他方法用同一 `p0_paper_eval.py` 组合评估。
`Qwen + Guard` 是混合算法，Guard 读取完整物理场景信息，
因此与仅使用原有 MARL 观测的 MAPPO 并非严格同信息条件。

## 八、应严格保留的研究结论边界

1. 五个小样本用于代码调试，不可当成论文总体性能。
2. 论文 83.98% 对应原作者的评估配置；我们的前 512 场景只能做本地同样本数比较。
3. 固定数值 + Guard ≠ ERA，Qwen + Guard ≠ Qwen-only。
4. 需要同时报告接纳率、总效用、新旧切片效用、成本、补偿及个体理性。
5. 强化学习/非强化学习之间的相同信息条件，仍需要做单独消融。
6. Guard 无法保证所有场景都有可行解；不会为了漂亮结果静默剔除拒绝样本。
7. P0-v5 的新增脚本尚未在你个人 Windows/XuanCe 环境上完成端到端 512 场景实测；
   离线统计器已通过真实历史记录核对，但完整运行器需要你本地验证。

版本：P0-v5 (2026-10-14)。
