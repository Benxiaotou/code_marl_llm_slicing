"""P0-v3b: add bandwidth-capacity pressure features derived from the original observations.

Prerequisite: P0-v3a installed. Run beside p0_policy.py.
Modifies only p0_policy.py: g link-description features and system prompt.
Does NOT modify original GEANT state, reward, transitions, MAPPO, or action parsing.
"""

from pathlib import Path
import hashlib
import os
import shutil

OLD_LINKS = '''"links": [{"i": i, "physical_bandwidth_allocated_ratio": _float(row[1]),
                           "requested_bandwidth_ratio": _float(row[2])}
                          for i, row in enumerate(links)
                          if row[1] != 0 or row[2] != 0],'''
NEW_LINKS = '''"links": [{
                    "i": i,
                    "physical_bandwidth_allocated_ratio": _float(row[1]),
                    "requested_bandwidth_ratio": _float(row[2]),
                    "free_bandwidth_ratio_before_k_release": _float(1.0 - row[1]),
                    "bandwidth_shortfall_ratio_before_k_release":
                        _float(max(0.0, row[1] + row[2] - 1.0)),
                    "min_g_bw_reduction_if_no_k_link_release":
                        _float(max(0.0, (row[1] + row[2] - 1.0)
                                   / (0.7 * row[2]))) if row[2] > 1e-12 else 0.0,
                    "shortfall_after_max_g_bw_reduction_if_no_k_link_release":
                        _float(max(0.0, row[1] + 0.3 * row[2] - 1.0)),
                }
                for i, row in enumerate(links)
                if row[1] != 0 or row[2] != 0],'''

PROMPT_ANCHOR = "Use the overlap group and utilization/capacity observations to choose ratios."
PROMPT_INSERT = (
    "The g-link bandwidth-capacity pressure fields are DERIVED FROM THE SAME "
    "ORIGINAL OBSERVATIONS and refer to the state BEFORE any existing-slice "
    "link release. The min_g_bw_reduction_if_no_k_link_release field is "
    "a per-link lower bound, NOT a guaranteed feasible solution. If its value "
    "exceeds 1 for any g link, reducing g bandwidth alone cannot fit that link: "
    "some relevant existing-slice link-group release is necessary. If it is "
    "at most 1, increasing g's bandwidth reduction MAY help without k-link "
    "releases; also respect the <=0.7 link-utilization check. "
    "Do not give all existing-slice link groups zero by default while "
    "bandwidth shortages remain. Avoid excessive node releases once CPU/cache "
    "are plausible. Prefer the smallest joint releases that plausibly satisfy "
    "both NODE and LINK admission checks. "
)

MARKER = '"min_g_bw_reduction_if_no_k_link_release"'

def main():
    root = Path(__file__).resolve().parent
    target = root / "p0_policy.py"
    backup = root / "p0_policy.py.before_p0_v3b"
    temporary = root / "p0_policy.py.p0_v3b.part"
    if not target.is_file():
        raise SystemExit("ERROR: p0_policy.py not found beside this script")
    original = target.read_bytes()
    try:
        content = original.decode("utf-8")
    except UnicodeError as exc:
        raise SystemExit("ERROR: p0_policy.py cannot be decoded as UTF-8: {}".format(exc))

    if MARKER in content and OLD_LINKS not in content:
        print("P0-v3b already installed; no changes.")
        return
    if content.count(OLD_LINKS) != 1 or content.count(PROMPT_ANCHOR) != 1:
        raise SystemExit(
            "ERROR: unexpected policy version. P0-v3a must be installed. "
            "No file changed."
        )
    if "Feasibility rules of the ORIGINAL environment" not in content:
        raise SystemExit("ERROR: P0-v3a constraints prompt missing; no changes.")
    if backup.exists():
        raise SystemExit(
            "ERROR: backup already exists: {}. No file changed.".format(backup)
        )

    changed = content.replace(OLD_LINKS, NEW_LINKS, 1)
    changed = changed.replace(PROMPT_ANCHOR,
                              PROMPT_INSERT + PROMPT_ANCHOR, 1)
    compile(changed, str(target), "exec")

    shutil.copy2(str(target), str(backup))
    try:
        temporary.write_bytes(changed.encode("utf-8"))
        os.replace(str(temporary), str(target))
    except Exception:
        if temporary.exists():
            temporary.unlink()
        shutil.copy2(str(backup), str(target))
        raise
    print("Installed P0-v3b bandwidth-pressure input and prompt.")
    print("Backup:", backup)
    print("SHA256:", hashlib.sha256(target.read_bytes()).hexdigest())
    print("Rollback:")
    print(r"  Copy-Item .\p0_policy.py.before_p0_v3b .\p0_policy.py -Force")


if __name__ == "__main__":
    main()
