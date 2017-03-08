import numpy as np
import tensorflow as tf

from rltools import tfutil
from rltools import util

TINY = 1e-10


class Distribution(object):

    @property
    def dim(self):
        raise NotImplementedError()

    def kl(self, old, new):
        raise NotImplementedError()

    def log_density(self, dist_params, x):
        raise NotImplementedError()

    def entropy(self, logprobs_N_K):
        raise NotImplementedError()

    def sample(self, logprobs_N_K):
        raise NotImplementedError()

    def kl_expr(self, logprobs1, logprobs2):
        raise NotImplementedError()

    def log_density_expr(self, dist_params, x):
        raise NotImplementedError()


class Categorical(Distribution):

    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def log_density(self, dist_params_B_A, x_B_A):
        return util.lookup_last_idx(dist_params_B_A, x_B_A)

    def entropy(self, probs_N_K):
        tmp = -probs_N_K * np.log(probs_N_K + TINY)
        tmp[~np.isfinite(tmp)] = 0
        return tmp.sum(axis=1)

    def sample(self, probs_N_K):
        """Sample from N categorical distributions, each over K outcomes"""
        N, K = probs_N_K.shape
        return np.array([np.random.choice(K, p=probs_N_K[i, :]) for i in range(N)])

    def kl_expr(self, logprobs1_B_A, logprobs2_B_A, name=None):
        """KL divergence between categorical distributions, specified as log probabilities"""
        with tf.op_scope([logprobs1_B_A, logprobs2_B_A], name, 'categorical_kl') as scope:
            kl_B = tf.reduce_sum(
                tf.exp(logprobs1_B_A) * (logprobs1_B_A - logprobs2_B_A), 1, name=scope)
        return kl_B

    def log_density_expr(self, dist_params_B_A, x_B_A):
        """Log density from categorical distribution params"""
        return tfutil.lookup_last_idx(dist_params_B_A, x_B_A)


class RecurrentCategorical(Distribution):

    def __init__(self, dim):
        self._dim = dim
        self._cat = Categorical(dim)

    @property
    def dim(self):
        return self._dim

    def log_density(self, dist_params_B_H_A, x_B_H_A):
        adim = dist_params_B_H_A.shape[-1]
        flat_logd = self._cat.log_density(
            dist_params_B_H_A.reshape((-1, adim)), x_B_H_A.reshape((-1, adim)))
        return flat_logd.reshape(dist_params_B_H_A.shape)

    def entropy(self, probs_N_H_K):
        tmp = -probs_N_H_K * np.log(probs_N_H_K + TINY)
        tmp[~np.isfinite(tmp)] = 0
        return tmp.sum(axis=-1)

    def sample(self, probs_N_K):
        """Sample from N categorical distributions, each over K outcomes"""
        return self._cat.sample(probs_N_K)

    def kl_expr(self, logprobs1_B_H_A, logprobs2_B_H_A, name=None):
        """KL divergence between categorical distributions, specified as log probabilities"""
        with tf.op_scope([logprobs1_B_H_A, logprobs2_B_H_A], name, 'categorical_kl') as scope:
            kl_B_H = tf.reduce_sum(
                tf.exp(logprobs1_B_H_A) * (logprobs1_B_H_A - logprobs2_B_H_A), 2, name=scope)
        return kl_B_H

    def log_density_expr(self, dist_params_B_H_A, x_B_H_A):
        adim = tf.shape(dist_params_B_H_A)[len(dist_params_B_H_A.get_shape()) - 1]
        flat_logd = self._cat.log_density_expr(
            tf.reshape(dist_params_B_H_A, tf.pack([-1, adim])),
            tf.reshape(x_B_H_A, tf.pack([-1, adim])))
        return tf.reshape(flat_logd, tf.shape(dist_params_B_H_A)[:2])


class Gaussian(Distribution):

    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def entropy(self, stdevs):
        d = stdevs.shape[-1]
        return .5 * d * (1. + np.log(2. * np.pi)) + np.log(stdevs).sum(axis=-1)

    def kl_expr(self, means1_stdevs1, means2_stdevs2, name=None):
        """KL divergence wbw diagonal covariant gaussians"""
        means1, stdevs1 = means1_stdevs1
        means2, stdevs2 = means2_stdevs2
        with tf.op_scope([means1, stdevs1, means2, stdevs2], name, 'gaussian_kl') as scope:
            D = tf.shape(means1)[len(means1.get_shape()) - 1]
            kl = tf.mul(.5, (tf.reduce_sum(tf.square(stdevs1 / stdevs2), -1) + tf.reduce_sum(
                tf.square((means2 - means1) / stdevs2), -1) + 2. * (tf.reduce_sum(
                    tf.log(stdevs2), -1) - tf.reduce_sum(tf.log(stdevs1), -1)) - tf.to_float(D)),
                        name=scope)
        return kl

    def log_density_expr(self, means, stdevs, x, name=None):
        """Log density of diagonal gauss"""
        with tf.op_scope([means, stdevs, x], name, 'gauss_log_density') as scope:
            D = tf.shape(means)[len(means.get_shape()) - 1]
            lognormconsts = -.5 * tf.to_float(D) * np.log(2. * np.pi) + 2. * tf.reduce_sum(
                tf.log(stdevs), -1)  # log norm consts
            logprobs = tf.add(-.5 * tf.reduce_sum(tf.square((x - means) / stdevs), -1),
                              lognormconsts, name=scope)
        return logprobs


RecurrentGaussian = Gaussian
