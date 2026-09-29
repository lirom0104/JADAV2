import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from utils.training_state import capture_runtime, restore_runtime, replay_camera_sampler


class RuntimeStateTests(unittest.TestCase):
    def test_legacy_camera_replay_crosses_multiple_epochs(self):
        cameras = list(range(7))
        random.seed(42)
        stack = []
        for _ in range(19):
            if not stack:
                stack = cameras.copy()
            stack.pop(random.randint(0,len(stack)-1))
        expected_state = random.getstate()
        random.seed(42)
        actual = replay_camera_sampler(cameras,19)
        self.assertEqual(actual,stack)
        self.assertEqual(random.getstate(),expected_state)

    def test_random_streams_camera_stack_and_ema_resume(self):
        cameras = [SimpleNamespace(uid=i) for i in (2,7,10)]
        model = SimpleNamespace(cmo_start_point_count=123,cmo_budget_reference_count=456)
        ema = {"weights": torch.tensor([.2,.3])}
        random.seed(11)
        np.random.seed(11)
        torch.manual_seed(11)
        with patch('torch.cuda.get_rng_state_all', return_value=[]), patch('torch.cuda.set_rng_state_all'):
            state = capture_runtime([cameras[2],cameras[0]],ema,model,11)
            expected = (random.random(),np.random.rand(),torch.rand(3))
            model.cmo_start_point_count, model.cmo_budget_reference_count = 0,0
            actual_stack, actual_ema = restore_runtime(state,cameras,model,11)
            actual = (random.random(),np.random.rand(),torch.rand(3))
        self.assertEqual([c.uid for c in actual_stack],[10,2])
        self.assertEqual((model.cmo_start_point_count,model.cmo_budget_reference_count),(123,456))
        torch.testing.assert_close(actual_ema['weights'],ema['weights'])
        self.assertEqual(actual[:2],expected[:2])
        torch.testing.assert_close(actual[2],expected[2])

    def test_seed_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError,'seed'):
            restore_runtime({'seed':1},[],None,2)


if __name__ == '__main__':
    unittest.main()
