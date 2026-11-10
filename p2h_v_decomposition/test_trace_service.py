import unittest
from trace_service import first_node_failure, path_load


class TraceSemantics(unittest.TestCase):
    def test_fixed_demand_changes_income_with_route_length(self):
        self.assertEqual(path_load(2., [[0, 1]]), 2.)
        self.assertEqual(path_load(2., [[0, 2, 1]]), 4.)

    def test_multipath_uses_equal_split_not_longest_route(self):
        self.assertEqual(path_load(3., [[0, 1], [0, 2, 3, 1]]), 6.)

    def test_first_failure_respects_node_then_resource_order(self):
        nodes = [
            {"local_index": 0, "compute_gap": 1., "cache_gap": -1.,
             "compute_physical_capacity": 10., "cache_physical_capacity": 20.},
            {"local_index": 1, "compute_gap": -5., "cache_gap": 0.,
             "compute_physical_capacity": 10., "cache_physical_capacity": 20.},
        ]
        f = first_node_failure(nodes)
        self.assertEqual((f["stage"], f["local_index"]), ("cache", 0))
        self.assertEqual(f["gap_fraction_of_physical"], -.05)


if __name__ == "__main__":
    unittest.main()
