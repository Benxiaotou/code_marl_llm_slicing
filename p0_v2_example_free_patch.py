"""P0-v2: remove the numeric output example from the existing system prompt.

Run next to p0_policy.py inside your Windows project root.
Changes exactly one prompt sentence. Backs up the original file.
Does not modify MAPPO, environment transitions, reward or action parsing.
"""
from pathlib import Path
import hashlib
import os
import shutil

OLD = (
    'Output exactly, for example: '
    '{"agent_k1":[0.2,0.3,0.2,0.3],"agent_k2":[0.2,0.3,0.2,0.3],'
    '"agent_k3":[0.2,0.3,0.2,0.3],"agent_g":[0.2,0.2,0.2]}.'
)
NEW = (
    'Choose all fifteen numeric ratios independently from the CURRENT scenario '
    'observations: the overlapping node/link control groups, existing-slice '
    'utilization and capacities, and new-slice resource pressures. '
    'Do not reuse a canned/default numeric vector or copy formatting examples. '
    'Different groups and slices may require different decisions; identical '
    'values are acceptable only if supported by the current data. '
    'Output only the required JSON object, containing numeric values in [0,1].'
)

def main():
    root = Path(__file__).resolve().parent
    target = root / "p0_policy.py"
    backup = root / "p0_policy.py.p0_v1_numeric_example_backup"
    temp = root / "p0_policy.py.p0_v2_tmp"
    if not target.is_file():
        raise SystemExit("ERROR: put this patch file beside p0_policy.py")
    original = target.read_bytes()
    try:
        source = original.decode("utf-8")
    except UnicodeError as exc:
        raise SystemExit("ERROR: p0_policy.py is not UTF-8: {}".format(exc))
    if NEW in source and OLD not in source:
        print("P0-v2 prompt is already installed; nothing changed.")
        return
    if source.count(OLD) != 1:
        raise SystemExit(
            "ERROR: expected example sentence not found exactly once. "
            "No file changed; inspect your local p0_policy.py first."
        )
    if backup.exists():
        raise SystemExit(
            "ERROR: original backup already exists: {}. "
            "No file changed.".format(backup)
        )
    edited = source.replace(OLD, NEW, 1)
    # Catch accidental syntax mistakes without executing user code.
    compile(edited, str(target), "exec")
    shutil.copy2(str(target), str(backup))
    try:
        temp.write_bytes(edited.encode("utf-8"))
        os.replace(str(temp), str(target))
    except Exception:
        if temp.exists():
            temp.unlink()
        shutil.copy2(str(backup), str(target))
        raise
    print("Installed P0-v2 example-free prompt.")
    print("Only SYSTEM_PROMPT example sentence was replaced.")
    print("Original backup:", backup)
    print("Updated SHA256:", hashlib.sha256(target.read_bytes()).hexdigest())
    print("Rollback (PowerShell): Copy-Item .\\p0_policy.py.p0_v1_numeric_example_backup .\\p0_policy.py -Force")

if __name__ == "__main__":
    main()
