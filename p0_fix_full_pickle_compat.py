# -*- coding: utf-8 -*-
"""Fix trusted ORIGINAL 600-scene GEANT pickle's legacy __main__ class reference.

Run from your Windows project root, beside p0_runner.py and
p0_guard_runner_v5.py. Does not change main.py, any samples,
the environment, rewards, MAPPO, or policy/guard math.

Reason: the old 600-scene pickle stores a reference to
__main__.InitVnetworkTrain, but newer entrypoint scripts are __main__.
"""
from pathlib import Path
import os
import sys

MARKER = b"# P0_FULL_PICKLE_LEGACY_ALIAS_V1"
BACKUP_SUFFIX = ".before_p0_full_pickle_compat"
TARGETS = ("p0_runner.py", "p0_guard_runner_v5.py")
OPTIONAL_TARGET = "p0_guard_runner.py"

def prepare_patch(path):
    raw = path.read_bytes()
    if MARKER in raw:
        if b'sys.modules["__main__"].InitVnetworkTrain = InitVnetworkTrain' not in raw:
            raise RuntimeError("Patch marker found but alias missing: " + str(path))
        return None
    # Match one exact line to avoid changing unrelated imports or original env.
    nl = b"\r\n" if b"\r\n" in raw else b"\n"
    anchor = b"    from main import CustomMultiAgentEnv" + nl
    if raw.count(anchor) != 1:
        raise RuntimeError(
            "Unexpected version (cannot find unique class import) in: " + str(path))
    replacement = (
        b"    from main import CustomMultiAgentEnv, InitVnetworkTrain" + nl
        + b"    " + MARKER + nl
        + b"    # Original dataset was serialized with main.py executed as __main__." + nl
        + b'    sys.modules["__main__"].InitVnetworkTrain = InitVnetworkTrain' + nl
    )
    updated = raw.replace(anchor, replacement, 1)
    compile(updated, str(path), "exec")
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if backup.exists():
        raise RuntimeError("Backup already exists (refusing to overwrite): " + str(backup))
    return raw, updated, backup

def main():
    base = Path(__file__).resolve().parent
    candidates = [base / name for name in TARGETS]
    candidates += [base / OPTIONAL_TARGET] if (base / OPTIONAL_TARGET).exists() else []
    missing = [str(p) for p in candidates[:2] if not p.is_file()]
    if missing:
        raise SystemExit("Missing required runner(s), put this script in project root: " + ", ".join(missing))

    # Validate every file BEFORE modifying any of them.
    plans = []
    for path in candidates:
        result = prepare_patch(path)
        if result is None:
            print("Already fixed:", path.name)
        else:
            plans.append((path, *result))

    done = []
    try:
        for path, raw, updated, backup in plans:
            with backup.open("xb") as f:
                f.write(raw)
            tmp = path.with_name(path.name + ".p0_pickle_part")
            tmp.write_bytes(updated)
            os.replace(str(tmp), str(path))
            done.append((path, backup))
            print("Fixed:", path.name, "| backup:", backup.name)
    except Exception:
        # Revert any changed runners if patching another file failed.
        for path, backup in reversed(done):
            path.write_bytes(backup.read_bytes())
        raise

    print("Installed legacy pickle compatibility. main.py and datasets are unchanged.")
    print("For failed output directories, choose NEW --output-dir names on retry.")

if __name__ == "__main__":
    main()
