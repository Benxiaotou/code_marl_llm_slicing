import json
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import p2b_coverage as a

class CoverageTests(unittest.TestCase):
    def make_run(self, dest, dataset='val-a', delta=0):
        dest.mkdir()
        manifest = {'eval_dataset_id': dataset, 'eval_dataset_sha256': 'abc123',
                    'scene_ids': list(range(8)), 'method': 'ir_rerank'}
        (dest/'manifest.json').write_text(json.dumps(manifest), encoding='utf8')
        with (dest/'records.jsonl').open('w',encoding='utf8') as f:
            for i in range(8):
                row={'status':'ok','original_scene_id':i,'metrics':{
                   'cooperative_valid': bool(i%2), 'g_done': True, 'V':float(300+i+delta),
                   'cost_sum':float(30+i-delta)},
                   'per_role':{r:{'distances_for_audit_only': [0.2+i*.01-delta*.001]}
                            for r in a.ROLES}}
                f.write(json.dumps(row)+'\n')
    def test_summary_and_pairing(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t);self.make_run(d/'a');self.make_run(d/'b',delta=1)
            ma,ra,sa=a.load_run(d/'a');mb,rb,sb=a.load_run(d/'b')
            s=a.summarize(ma,ra,sa)
            self.assertEqual(s['n'],8)
            self.assertEqual(s['cooperative_valid_count'],4)
            self.assertAlmostEqual(a.compare(ma,ra,mb,rb)['mean_V_delta'],1)
            self.assertLess(a.compare(ma,ra,mb,rb)['roles']['agent_k1']['paired_nearest_mean_delta'],0)
    def test_reject_mismatched_dataset(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t);self.make_run(d/'a');self.make_run(d/'b',dataset='val-b')
            ma,ra,_=a.load_run(d/'a');mb,rb,_=a.load_run(d/'b')
            with self.assertRaises(ValueError): a.compare(ma,ra,mb,rb)
    def test_reject_fallback(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t);self.make_run(d/'a')
            p=d/'a'/'records.jsonl'
            ss=p.read_text().splitlines();o=json.loads(ss[0]);o['fallback_used']=True
            ss[0]=json.dumps(o);p.write_text('\n'.join(ss)+'\n')
            with self.assertRaises(ValueError): a.load_run(d/'a')

if __name__ == '__main__': unittest.main()
