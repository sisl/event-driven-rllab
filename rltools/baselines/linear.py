import numpy as np
import scipy.linalg
import tensorflow as tf

from rltools import nn
from rltools.baselines import Baseline


class LinearFeatureBaseline(Baseline):

    def __init__(self, observation_space, enable_obsnorm, reg_coeff=1e-5, varscope_name='linear'):
        super(LinearFeatureBaseline, self).__init__(observation_space)
        self.w_Df = None
        self._reg_coeff = reg_coeff
        with tf.variable_scope(varscope_name + '_obsnorm'):
            self.obsnorm = (nn.Standardizer if enable_obsnorm else
                            nn.NoOpStandardizer)(self.observation_space.shape[0])

    def get_params(self, _):
        return self.w_Df

    def set_params(self, _, vals):
        self.w_Df = vals

    def update_obsnorm(self, obs_B_Do, sess):
        """Update norms using moving avg"""
        self.obsnorm.update(obs_B_Do, sess=sess)

    def _features(self, sess, trajs):
        obs_B_Do = trajs.obs.stacked
        sobs_B_Do = self.obsnorm.standardize(obs_B_Do)
        return np.concatenate([
            sobs_B_Do.reshape(len(sobs_B_Do), np.prod(sobs_B_Do.shape[1:])), 
            trajs.time.stacked[:, None] / 100., (trajs.time.stacked[:, None] / 100.)**2,
            np.ones((sobs_B_Do.shape[0], 1))
        ], axis=1)

    def fit(self, sess, trajs, qvals):
        assert qvals.shape == (trajs.obs.stacked.shape[0],)
        feat_B_Df = self._features(sess, trajs)
        self.w_Df = scipy.linalg.solve(
            feat_B_Df.T.dot(feat_B_Df) + self._reg_coeff * np.eye(feat_B_Df.shape[1]),
            feat_B_Df.T.dot(qvals), sym_pos=True)
        return []

    def predict(self, sess, trajs):
        feat_B_Df = self._features(sess, trajs)
        if self.w_Df is None:
            self.w_Df = np.zeros(feat_B_Df.shape[1], dtype=trajs.obs.stacked.dtype)
        return feat_B_Df.dot(self.w_Df)
