import configparser
import unittest

from ga_lem_inverter.pipeline.forward_model import boundary_status_from_config, normalize_boundary_status


class BoundaryConfigAcceptanceTests(unittest.TestCase):
    def test_scalar_boundary_status_still_works(self):
        """产品验收：旧版 boundary_status=fixed_value 仍然可用。"""
        cfg = configparser.ConfigParser()
        cfg["Model"] = {"boundary_status": "fixed_value"}
        self.assertEqual(boundary_status_from_config(cfg), "fixed_value")

    def test_per_edge_boundary_status_sets_downstream_outlet(self):
        """产品验收：用户可把下游边界设为出水口，其余边界闭合。"""
        cfg = configparser.ConfigParser()
        cfg["Model"] = {
            "boundary_status": "fixed_value",
            "boundary_left": "fixed_value",
            "boundary_right": "fixed_value",
            "boundary_top": "fixed_value",
            "boundary_bottom": "core",
        }
        self.assertEqual(boundary_status_from_config(cfg), ["fixed_value", "fixed_value", "fixed_value", "core"])

    def test_comma_boundary_status_uses_fastscape_border_order(self):
        """产品验收：逗号写法按 left,right,top,bottom 顺序解析。"""
        self.assertEqual(
            normalize_boundary_status("fixed_value,fixed_value,fixed_value,core"),
            ["fixed_value", "fixed_value", "fixed_value", "core"],
        )

    def test_invalid_boundary_status_fails_early(self):
        """产品验收：边界配置拼错时必须在进入长时间优化前报错。"""
        with self.assertRaisesRegex(ValueError, "无效边界状态"):
            normalize_boundary_status(["fixed_value", "bad", "fixed_value", "core"])


if __name__ == "__main__":
    unittest.main()
