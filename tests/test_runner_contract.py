import unittest

from ga_lem_inverter.runner import WORKFLOWS, _load_workflow
from ga_lem_inverter.workflows.k_sensitivity import KSensitivityExperiment
from ga_lem_inverter.workflows.synthetic import SyntheticExperiment


class RunnerContractAcceptanceTests(unittest.TestCase):
    def test_all_public_modes_resolve_through_runner_registry(self):
        """产品验收：用户可见模式必须都由 runner 注册表解析，避免误走旧实验类。"""
        self.assertEqual(set(WORKFLOWS), {"main", "synthetic", "k_sensitivity", "pecube_coupled"})
        for mode in WORKFLOWS:
            workflow = _load_workflow(mode)
            self.assertTrue(callable(workflow), mode)

    def test_legacy_experiment_classes_are_marked_deprecated(self):
        """产品验收：旧类仍兼容，但实例化时必须提示用户改用 runner/config。"""
        with self.assertWarns(DeprecationWarning):
            SyntheticExperiment()
        with self.assertWarns(DeprecationWarning):
            KSensitivityExperiment()


if __name__ == "__main__":
    unittest.main()
