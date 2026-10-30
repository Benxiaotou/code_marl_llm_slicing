import json
import tempfile
import unittest
from pathlib import Path

from p1a_compare import compare


def source(scene):
    data = {"g_done": True, "V": 320.0, "V_g": 20.0, "cost_sum": 12.0,
            "V_k1": 100.0, "V_k2": 100.0, "V_k3": 100.0,
            "V_k1_0": 99.0, "V_k2_0": 99.0, "V_k3_0": 99.0}
    p1 = {"status": "ok", "original_scene_id": scene, "metrics": data, "fallback_used": False}
    p0 = {"original_scene_index": scene, "used_fallback": False, "decision_error": None}
    p0.update(data)
    p0["V"] = 315.0
    return p1, p0


class CompareTests(unittest.TestCase):
    def test_match_and_wrong_scene_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / 'a.jsonl', Path(d) / 'b.jsonl'
            p1, p0 = source(299)
            a.write_text(json.dumps(p1) + '\n', encoding='utf8')
            b.write_text(json.dumps(p0) + '\n', encoding='utf8')
            summary, rows = compare(a, b)
            self.assertEqual(summary['paired_delta_V'], 5)
            self.assertEqual(rows[0]['scene_id'], 299)
            p0['original_scene_index'] = 599
            b.write_text(json.dumps(p0) + '\n', encoding='utf8')
            with self.assertRaises(ValueError):
                compare(a, b)

    def test_failure_cannot_be_presented_as_zero(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / 'a.jsonl', Path(d) / 'b.jsonl'
            p1, p0 = source(0)
            p1['status'] = 'policy_error'
            a.write_text(json.dumps(p1) + '\n', encoding='utf8')
            b.write_text(json.dumps(p0) + '\n', encoding='utf8')
            with self.assertRaises(ValueError):
                compare(a, b)


if __name__ == '__main__':
    unittest.main()
