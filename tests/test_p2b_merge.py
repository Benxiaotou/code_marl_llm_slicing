import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import p2b_merge as m


def write_source(root, dataset, seed, count, strata, *, identical_ids=False):
    root.mkdir(parents=True, exist_ok=True)
    (root / "private_offline").mkdir()
    (root / "role_libraries").mkdir()
    mf = {
        "schema": "p1b_library_manifest_v1", "dataset_id": dataset,
        "training_sample_sha256": f"{seed:064x}",
        "teacher_provenance": {"verified_mappo": True, "kind": "live_mappo",
                               "checkpoint_sha256": "f" * 64},
        "development_only": False, "n_experiences_per_role": count,
        "roles": list(m.ROLES), "no_current_global_information": True,
        "output_role_fields": list(sorted(m.ROLE_KEYS)),
        "warnings": ["Original joint cases are positive examples"],
    }
    m.emit_json(root / "library_manifest.json", mf)
    m.emit_json(root / "audit_report.json", {
        "audit_passed": True, "dataset_id": dataset,
        "n_per_role": count, "roles": list(m.ROLES),
    })
    joint = []
    roles = {r: [] for r in m.ROLES}
    for i in range(count):
        ident = (f"same-{i}" if identical_ids else f"{dataset}:scene{i}")
        obs = {r: [round(seed / 100.0 + i / 1000.0 + k / 10000.0 + j / 1000000.0, 7)
                   for j in range(240)]
               for k, r in enumerate(m.ROLES)}
        ratio = {r: [0.2] * (3 if r == "agent_g" else 4) for r in m.ROLES}
        env = {r: [-0.6] * (3 if r == "agent_g" else 4) for r in m.ROLES}
        item = {
            "status": "ok", "dataset_id": dataset, "joint_id": ident,
            "observations": obs, "teacher_actions_ratio": ratio,
            "teacher_actions_env": env,
            "metrics": {"cooperative_valid": True, "g_done": True},
            "selection_quality_score": (count - i) / 100,
            "selection_stratum": strata[i % len(strata)],
            "offline_pressure": float(i % 5), "offline_overlap": float(i % 2),
        }
        joint.append(item)
        for r in m.ROLES:
            roles[r].append({
                "schema": "p1b_local_case_v1", "role": r,
                "experience_id": ident, "observation_240": obs[r],
                "own_teacher_ratios": ratio[r],
                "own_teacher_env_action": env[r],
                "source": "frozen_MAPPO_selected_joint_cooperation",
            })
    m.emit_jsonl(root / "private_offline" / "selected_joint_records.jsonl", joint)
    for r in m.ROLES:
        m.emit_jsonl(root / "role_libraries" / f"{r}.jsonl", roles[r])
    return root


class P2BMergeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old = write_source(self.root / "old", "train1", 1, 3, ["low", "medium"])
        self.new = write_source(self.root / "new", "train2", 2, 9, ["low", "low", "low", "high"])
        self.holdout = self.root / "val.json"
        m.emit_json(self.holdout, {"dataset_id": "val", "split": "validation",
                                   "sample_sha256": "e" * 64})

    def make_args(self, execute=False):
        from argparse import Namespace
        return Namespace(old_library=str(self.old), new_library=str(self.new),
                         holdout_manifest=[str(self.holdout)],
                         output_dir=str(self.root / "out"),
                         sizes="5,8,10", expect_old=3, expect_new=9,
                         execute=execute)

    def test_plan_is_read_only(self):
        result = m.run(self.make_args(False))
        self.assertFalse(result["executed"])
        self.assertEqual(result["pool_count"], 12)
        self.assertEqual(len(result["outputs"]), 6)
        self.assertFalse((self.root / "out").exists())

    def test_build_nested_roles_and_old_anchor(self):
        old_hash = m.sha_file(self.old / "library_manifest.json")
        new_hash = m.sha_file(self.new / "library_manifest.json")
        result = m.run(self.make_args(True))
        self.assertTrue(result["executed"])
        self.assertEqual(len(result["outputs"]), 6)
        self.assertEqual(m.sha_file(self.old / "library_manifest.json"), old_hash)
        self.assertEqual(m.sha_file(self.new / "library_manifest.json"), new_hash)
        for mode in ("standard", "balanced"):
            previous = None
            for count in (5, 8, 10):
                path = self.root / "out" / f"p2b_{mode}_{count}_v1"
                joint = m.read_jsonl(path / "private_offline" / "selected_joint_records.jsonl")
                ids = [r["joint_id"] for r in joint]
                self.assertEqual(len(joint), count)
                self.assertEqual(ids[:3], [f"train1:scene{i}" for i in range(3)])
                if previous is not None:
                    self.assertEqual(ids[:len(previous)], previous)
                previous = ids
                meta = m.read_json(path / "library_manifest.json")
                self.assertEqual(meta["n_experiences_per_role"], count)
                self.assertEqual(len(meta["source_training_datasets"]), 2)
                for role in m.ROLES:
                    data = m.read_jsonl(path / "role_libraries" / f"{role}.jsonl")
                    self.assertEqual([r["experience_id"] for r in data], ids)
                    for r in data:
                        self.assertEqual(set(r), m.ROLE_KEYS)
        # The balanced strategy uses source P1-B strata rather than merely copying new order.
        a = m.read_jsonl(self.root / "out" / "p2b_standard_8_v1" /
                         "private_offline" / "selected_joint_records.jsonl")
        b = m.read_jsonl(self.root / "out" / "p2b_balanced_8_v1" /
                         "private_offline" / "selected_joint_records.jsonl")
        self.assertNotEqual([r["joint_id"] for r in a], [r["joint_id"] for r in b])

    def test_holdout_source_overlap_rejected(self):
        m.emit_json(self.holdout, {"dataset_id": "val", "split": "validation",
                                   "sample_sha256": "1".zfill(64)})
        with self.assertRaises(m.InputError):
            m.run(self.make_args(False))

    def test_identity_collision_rejected(self):
        new = m.read_jsonl(self.new / "private_offline" / "selected_joint_records.jsonl")
        new[0]["joint_id"] = "train1:scene0"
        m.emit_jsonl(self.new / "private_offline" / "selected_joint_records.jsonl", new)
        for role in m.ROLES:
            p = self.new / "role_libraries" / f"{role}.jsonl"
            data = m.read_jsonl(p)
            data[0]["experience_id"] = "train1:scene0"
            m.emit_jsonl(p, data)
        with self.assertRaises(m.InputError):
            m.run(self.make_args(False))

    def test_source_audit_detects_mismatch(self):
        p = self.new / "role_libraries" / "agent_k1.jsonl"
        rows = m.read_jsonl(p)
        rows[0]["observation_240"][2] = 123.45
        m.emit_jsonl(p, rows)
        with self.assertRaises(m.InputError):
            m.run(self.make_args(False))

    def test_cross_source_duplicate_observation_rejected(self):
        old = m.read_jsonl(self.old / "private_offline" / "selected_joint_records.jsonl")
        new_path = self.new / "private_offline" / "selected_joint_records.jsonl"
        new = m.read_jsonl(new_path)
        new[0]["observations"] = old[0]["observations"]
        m.emit_jsonl(new_path, new)
        for role in m.ROLES:
            p = self.new / "role_libraries" / f"{role}.jsonl"
            data = m.read_jsonl(p)
            data[0]["observation_240"] = old[0]["observations"][role]
            m.emit_jsonl(p, data)
        with self.assertRaises(m.InputError):
            m.run(self.make_args(False))

    def test_unknown_role_field_rejected(self):
        p = self.new / "role_libraries" / "agent_k2.jsonl"
        data = m.read_jsonl(p)
        data[0]["other_role_reward"] = 3.5
        m.emit_jsonl(p, data)
        with self.assertRaises(m.InputError):
            m.run(self.make_args(False))

    def test_refuse_overwrite(self):
        m.run(self.make_args(True))
        with self.assertRaises(FileExistsError):
            m.run(self.make_args(True))


if __name__ == "__main__":
    unittest.main()
