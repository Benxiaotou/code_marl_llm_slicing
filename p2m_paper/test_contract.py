import copy
import unittest
import numpy as np
import common as c


class SupplementContracts(unittest.TestCase):
    def setUp(self):
        obs=np.zeros(240); obs[:60].reshape(12,5)[:]=[1,.3,.7,.2,.8]
        self.obs={r:obs.copy() for r in c.v.ROLES}
        self.raw={r:[0.,0.,-.4,.4] for r in c.v.ROLES}
        self.targets={r:[.8,.2] for r in c.v.K}

    def test_factorial_orthogonality_immutability_and_caps(self):
        before=copy.deepcopy(self.raw); modes=c.variants(self.raw,self.obs,self.targets)
        self.assertEqual(before,self.raw)
        for r in c.v.K:
            self.assertEqual(modes['raw'][r],self.raw[r])
            self.assertEqual(modes['node'][r][2:],self.raw[r][2:])
            self.assertEqual(modes['link'][r][:2],self.raw[r][:2])
            self.assertEqual(modes['both'][r][:2],modes['node'][r][:2])
            self.assertEqual(modes['both'][r][2:],modes['link'][r][2:])
            self.assertTrue(all(0<=modes['node'][r][j]-self.raw[r][j]<=.050000001 for j in (0,1)))
        for mode in modes.values():
            self.assertEqual(mode['agent_g'],self.raw['agent_g'])

    def test_no_cross_policy_action_coupling(self):
        bases={p:copy.deepcopy(self.raw) for p in c.POLICIES}
        first=c.actions(bases,self.obs,self.targets)
        bases['MAPPO']['agent_k1'][0]=.5
        second=c.actions(bases,self.obs,self.targets)
        self.assertEqual(len(first),12)
        for p in ('kNN','Qwen'):
            for m in c.MODES:
                self.assertEqual(first[p+'_'+m],second[p+'_'+m])

    def test_knn_matches_original_policy(self):
        from p1c.policy import RetrievalPolicy
        class Library:
            cases={r:[{}, {}, {}] for r in c.v.ROLES}
            def query(self,role,observation,top_k=1):
                return [{'ratios':[.2,.3,.4] if role=='agent_g' else [.2,.3,.4,.5],
                         'experience_id':'fake_train','distance':0.}]
        lib=Library(); policy=RetrievalPolicy(lib,'knn',examples=3,seed=1); policy.set_scene(0)
        actual,_=policy.act(self.obs)
        self.assertEqual(c.knn_action(lib,self.obs),{r:a.tolist() for r,a in actual.items()})

    def test_new_test_design_and_claim_family_are_fixed(self):
        self.assertEqual(c.NEW_SEEDS,(2026110511,2026110512,2026110513))
        self.assertTrue(set(c.NEW_SEEDS).isdisjoint(c.OLD_SEEDS+(2026110501,2026110502)))
        self.assertEqual(c.TOTAL,1536); self.assertEqual(c.CALLS,6144)
        self.assertEqual(len(c.CLAIMS),3)
        self.assertEqual(c.CLAIMS['matched_MAPPO'],('Qwen_both','MAPPO_both'))


if __name__=='__main__':
    unittest.main()
