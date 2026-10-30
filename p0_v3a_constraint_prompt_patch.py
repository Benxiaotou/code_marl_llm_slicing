"""P0-v3a: tell Qwen the ORIGINAL environment's admission rules.

Runs beside p0_policy.py after installing the P0-v2 example-free patch.
Changes only one SYSTEM_PROMPT paragraph. No environment, MAPPO, reward,
observation representation, action parser, or network code is modified.
"""
from pathlib import Path
import hashlib
import os
import shutil

OLD = (
    'Choose all fifteen numeric ratios independently from the CURRENT scenario '
    'observations: the overlapping node/link control groups, existing-slice '
    'utilization and capacities, and new-slice resource pressures. '
    'Do not reuse a canned/default numeric vector or copy formatting examples. '
    'Different groups and slices may require different decisions; identical '
    'values are acceptable only if supported by the current data. '
    'Output only the required JSON object, containing numeric values in [0,1].'
)

NEW = """Choose the fifteen numeric ratios based on THIS scenario, not a canned vector.
Feasibility rules of the ORIGINAL environment (important, not generic assumptions):
1. For an existing slice k, a node-group action ratio r releases 0.8*r of
   that slice's previously allocated CPU AND cache capacity at its controlled
   overlapping nodes. A link-group action ratio r likewise releases 0.8*r
   of the controlled bandwidth capacities. Larger releases can cause migration
   or rerouting costs. The two groups use the indices supplied below.
2. For a new-slice g node, the ORIGINAL environment checks available physical
   CPU/cache against the ORIGINAL g CPU/cache requests before adding the
   reduced g allocation. Thus increasing g's CPU/cache reduction ratios does
   NOT relax this implementation's node-admission checks. To make g feasible,
   the existing slices must free enough physical CPU and cache at the relevant
   overlapping nodes. Do not assume all g node indices match k node indices.
3. In the new-slice observation, physical_*_allocated_ratio means occupied
   capacity, NOT free capacity. Before existing slices release resources,
   occupied_ratio + requested_ratio > 1 indicates a normalized CPU/cache
   shortfall at that g node. Existing-slice node releases can close it.
4. For g bandwidth, effective requested bandwidth is original*(1-0.7*r_g_bw).
   More reduction may help fit physical link capacity, BUT resulting new-slice
   link utilization must be at most 0.7 or admission still fails.
Decision priorities: (a) find a plausible joint action that enables g admission,
especially by addressing positive CPU/cache pressure; (b) avoid unnecessary
migration, rerouting, and bandwidth reduction; (c) preserve system utility.
Use the overlap group and utilization/capacity observations to choose ratios.
Do not make all values equal by default; do not invent exact physical-node
mappings or claim to have proved feasibility from the limited observations.
Return ONLY the required JSON object, with finite numeric values in [0,1],
exactly four numbers for each k agent and three for g. No example values,
no explanations, no Markdown."""

MARKER = "Feasibility rules of the ORIGINAL environment (important, not generic assumptions):"


def main():
    root = Path(__file__).resolve().parent
    target = root / "p0_policy.py"
    backup = root / "p0_policy.py.before_p0_v3a"
    temporary = root / "p0_policy.py.p0_v3a.part"
    if not target.is_file():
        raise SystemExit("ERROR: p0_policy.py not found next to this script")
    source_bytes = target.read_bytes()
    try:
        source = source_bytes.decode("utf-8")
    except UnicodeError as exc:
        raise SystemExit("ERROR: policy is not UTF-8: {}".format(exc))
    if MARKER in source and OLD not in source:
        print("P0-v3a already installed; no changes.")
        return
    if source.count(OLD) != 1:
        raise SystemExit(
            "ERROR: expected P0-v2 prompt paragraph not found exactly once. "
            "No changes made."
        )
    if backup.exists():
        raise SystemExit(
            "ERROR: backup already exists; no changes made: {}".format(backup)
        )
    new_source = source.replace(OLD, NEW, 1)
    compile(new_source, str(target), "exec")
    shutil.copy2(str(target), str(backup))
    try:
        temporary.write_bytes(new_source.encode("utf-8"))
        os.replace(str(temporary), str(target))
    except Exception:
        if temporary.exists():
            temporary.unlink()
        shutil.copy2(str(backup), str(target))
        raise
    print("Installed P0-v3a constraints-aware SYSTEM_PROMPT.")
    print("Original code backed up at:", backup)
    print("Updated SHA256:", hashlib.sha256(target.read_bytes()).hexdigest())
    print("Rollback:")
    print("  Copy-Item .\\p0_policy.py.before_p0_v3a .\\p0_policy.py -Force")


if __name__ == "__main__":
    main()
