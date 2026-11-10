"""从原版 GEANT 大型样本文件中安全导出少量真实测试场景。
请在项目根目录、原 MAPPO Python 环境中执行。仅处理自己信任的 pickle 文件。
"""
import argparse
import importlib
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

class ProjectUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            project_main = importlib.import_module("main")
            if hasattr(project_main, name):
                return getattr(project_main, name)
        return super().find_class(module, name)

def export_subset(source, output, indices):
    source, output = Path(source), Path(output)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.resolve() == output.resolve():
        raise ValueError("源文件和输出文件不能相同")
    if not indices or len(set(indices)) != len(indices) or min(indices) < 0:
        raise ValueError("索引必须为不重复的非负整数")

    print("正在读取原始数据，可能需要较大内存……", flush=True)
    with source.open("rb") as f:
        samples = ProjectUnpickler(f).load()
    if not isinstance(samples, dict):
        raise TypeError("原始数据不是样本字典")
    absent = [i for i in indices if i not in samples]
    if absent:
        raise KeyError(f"原始样本中不存在这些索引: {absent}")

    subset = {j: samples[i] for j, i in enumerate(indices)}
    if any(not isinstance(x, (tuple, list)) or len(x) != 5 for x in subset.values()):
        raise ValueError("单个样本结构不是预期的 (k1, k2, k3, g, physical)")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    try:
        with temporary.open("wb") as f:
            pickle.dump(subset, f, protocol=pickle.HIGHEST_PROTOCOL)
        with temporary.open("rb") as f:
            checked = ProjectUnpickler(f).load()
        if list(checked.keys()) != list(subset.keys()):
            raise RuntimeError("导出文件回读校验失败")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()

    index_file = output.with_suffix(".indices.json")
    index_file.write_text(
        json.dumps({str(j): i for j, i in enumerate(indices)}, indent=2),
        encoding="utf-8"
    )
    print(f"导出成功：{output}（{output.stat().st_size} 字节）")
    print(f"原始索引映射：{index_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/GEANT/GEANT_3_12_600_sample_save.pkl")
    parser.add_argument("--output", default="P0_small_samples.pkl")
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2, 299, 599])
    args = parser.parse_args()
    export_subset(args.source, args.output, args.indices)
