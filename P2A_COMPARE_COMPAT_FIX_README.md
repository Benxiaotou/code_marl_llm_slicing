# P2-A 比较器兼容性修复

问题：P1-D 原版 manifest 没有 `ir_risk_prompt_present` 和 `ir_history_data_present` 两个字段。P2-A 比较器此前将“字段缺失”误判为“显式 False”。

修复：对旧版 P1-D，允许这两个字段缺失；若字段明确存在，则必须为 True。保留训练 IR 历史哈希、模型参数、当前局部状态和实际完整 Prompt 的逐场景严格核对。

安装：备份本地 `p2a_compare.py` 与 `tests/test_p2a.py`，把本 ZIP 解压到 `F:\Projects\code_marl_v0_exper_1` 根目录。

验证：
```
python -m unittest discover -s tests -p "test_p2a.py" -v
```

然后重跑之前失败的 `python p2a_compare.py ...` 命令，将 `--output-dir` 换成 `.\p2a_runs\compare_val200_all9_fixed`。

不必重新运行 200 场景的 Qwen。已使用上传的真实 P2-A 和原始 P1-D 日志完成 200 场景兼容性与严格 Prompt/检索核验；67 项离线及回归测试通过。
