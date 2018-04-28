import unittest
import math
from models.sc2_model import Sc2Model


class DummyModel(Sc2Model):
    def _update(self, replay_buffer):
        return {}

    def _select_action(self, obs, illegal_internal_action_ids=None):
        pass


class TestBaseMethods(unittest.TestCase):

    def test_negative_lr_raises(self):
        """ Negative lr makes no sense and should raise an exception """
        with self.assertRaises(ValueError):
            DummyModel([], lr=-1)

    def test_lr_decay_goes_to_zero(self):
        """ After going through all the episodes the lr should have reached zero """
        episodes = 100
        model = DummyModel([], lr=0.01, decay_lr=True, total_episodes=episodes)

        for i in range(episodes):
            model.update([])

        self.assertAlmostEqual(model.lr, 0, 7)

    def test_lr_decay_halfs(self):
        """ After going through half the episodes the lr should have halved"""
        episodes = 100
        lr = 0.01
        model = DummyModel([], lr=lr, decay_lr=True, total_episodes=episodes)

        for i in range(int(episodes/2)):
            model.update([])

        difference = math.fabs(model.lr - lr/2)
        self.assertAlmostEqual(difference, 0, 7)


if __name__ == '__main__':
    unittest.main()
