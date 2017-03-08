import numpy as np

from rltools.baselines import Baseline


class ZeroBaseline(Baseline):

    def __init__(self, observation_space):
        pass

    def get_params(self, sess):
        return None

    def set_params(self, sess, val):
        pass

    def fit(self, sess, trajs, qvals):
        return []

    def predict(self, sess, trajs):
        return np.zeros_like(trajs.r.stacked)

    def update_obsnorm(self, sess, obs):
        pass
