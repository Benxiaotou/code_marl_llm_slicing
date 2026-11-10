import unittest
import numpy as np
from bounded_release import adjust_local


class LocalBoundsTest(unittest.TestCase):
    def obs(self, utils):
        obs = np.zeros(240)
        for i, u in enumerate(utils):
            obs[i * 5:i * 5 + 5] = [1, .3, u, .2, u]
        return obs

    def test_odd_overlap_groups_follow_repeat_and_worst_resource(self):
        obs = self.obs([.2, .2, .6, .2, .2])  # first group has3 nodes
        out, _ = adjust_local(obs, [0., 0., -.4, .6], .05, 0.)
        self.assertEqual(out[0], 0.)  # ratio.5 already at third-node limit
        self.assertAlmostEqual(out[1], .1)
        self.assertEqual(out[2:], [-.4, .6])

    def test_safe_does_not_add_to_existing_overload(self):
        out, _ = adjust_local(self.obs([.9, .9]), [0., 0., 0., 0.], .05, 0.)
        self.assertEqual(out, [0., 0., 0., 0.])

    def test_buffer_is_incremental_overload_not_global_capacity(self):
        out, audit = adjust_local(self.obs([.9, .2]), [0., 0., -.8, .7], .05, .02)
        self.assertAlmostEqual(out[0], .05, places=5)
        self.assertAlmostEqual(out[1], .1)
        self.assertLessEqual(audit[0]['max_extra_projected_overload'], .02)
        self.assertEqual(out[2:], [-.8, .7])

    def test_empty_groups_and_action_saturation(self):
        self.assertEqual(adjust_local(self.obs([]), [0., .8, .1, .2], .05, .02)[0],
                         [0., .8, .1, .2])
        out, _ = adjust_local(self.obs([0.]), [1., -.3, .1, .2], .05, .02)
        self.assertEqual(out, [1., -.3, .1, .2])

    def test_invalid_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            adjust_local(self.obs([.1]), [0.] * 4, .051, 0.)
        bad = self.obs([.1]); bad[0] = .5
        with self.assertRaises(ValueError):
            adjust_local(bad, [0.] * 4, .025, 0.)

    def test_random_bounds_for_both_resources(self):
        rng = np.random.default_rng(5601)
        for _ in range(200):
            obs = self.obs(rng.random(11))
            obs[:60].reshape(12, 5)[:11, 4] = rng.random(11)
            a = rng.uniform(-1, 1, 4)
            out, audit = adjust_local(obs, a, .05, .02)
            self.assertTrue(np.all(np.asarray(out[:2]) >= a[:2]))
            self.assertTrue(np.all(np.asarray(out[:2]) - a[:2] <= .100001))
            self.assertEqual(out[2:], a[2:].tolist())
            self.assertTrue(all(x['max_extra_projected_overload'] <= .02000001 for x in audit))


if __name__ == '__main__':
    unittest.main()
