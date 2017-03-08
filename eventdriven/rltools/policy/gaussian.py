from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from rltools import nn, tfutil
from rltools.distributions import Gaussian, RecurrentGaussian
from rltools.policy.stochastic import StochasticPolicy
from rltools.util import EzPickle


class GaussianMLPPolicy(StochasticPolicy, EzPickle):

    def __init__(self, observation_space, action_space, hidden_spec, enable_obsnorm, min_stdev,
                 init_logstdev, varscope_name):
        EzPickle.__init__(self, observation_space, action_space, hidden_spec, enable_obsnorm,
                          min_stdev, init_logstdev, varscope_name)
        self.hidden_spec = hidden_spec
        self.min_stdev = min_stdev
        self.init_logstdev = init_logstdev
        self._dist = Gaussian(action_space.shape[0])
        super(GaussianMLPPolicy, self).__init__(observation_space,
                                                action_space,
                                                action_space.shape[0] *
                                                2,  # Mean and diagonal stdev
                                                enable_obsnorm,
                                                varscope_name)

    @property
    def distribution(self):
        return self._dist

    def _make_actiondist_ops(self, obs_B_Df):
        with tf.variable_scope('flat'):
            flat = nn.FlattenLayer(obs_B_Df)
        with tf.variable_scope('hidden'):
            net = nn.FeedforwardNet(flat.output, flat.output_shape, self.hidden_spec)
        with tf.variable_scope('out'):
            mean_layer = nn.AffineLayer(net.output, net.output_shape, self.action_space.shape,
                                        Winitializer=tf.zeros_initializer, binitializer=None)

        means_B_Da = mean_layer.output

        # logstdev params
        logstdevs_1_Da = tf.get_variable('logstdevs_1_Da', shape=(1, self.action_space.shape[0]),
                                         initializer=tf.constant_initializer(self.init_logstdev))
        stdevs_1_Da = self.min_stdev + tf.exp(
            logstdevs_1_Da)  # Required for stability of kl computations
        stdevs_B_Da = tf.ones_like(means_B_Da) * stdevs_1_Da

        actiondist_B_Pa = tf.concat(1, [means_B_Da, stdevs_B_Da])
        return actiondist_B_Pa

    def _extract_actiondist_params(self, actiondist_B_Pa):
        means_B_Da = actiondist_B_Pa[:, :self.action_space.shape[0]]
        stdevs_B_Da = actiondist_B_Pa[:, self.action_space.shape[0]:]
        return means_B_Da, stdevs_B_Da

    def _make_actiondist_logprobs_ops(self, actiondist_B_Pa, input_actions_B_Da):
        means_B_Da, stdevs_B_Da = self._extract_actiondist_params(actiondist_B_Pa)
        return self.distribution.log_density_expr(means_B_Da, stdevs_B_Da, input_actions_B_Da)

    def _make_actiondist_kl_ops(self, proposal_actiondist_B_Pa, actiondist_B_Pa):
        return self.distribution.kl_expr(*map(self._extract_actiondist_params,
                                              [proposal_actiondist_B_Pa, actiondist_B_Pa]))

    def _sample_from_actiondist(self, actiondist_B_Pa, deterministic=False):
        means_B_Da, stdevs_B_Da = self._extract_actiondist_params(actiondist_B_Pa)
        if deterministic:
            return means_B_Da
        stdnormal_B_Da = np.random.randn(actiondist_B_Pa.shape[0], self.action_space.shape[0])
        assert stdnormal_B_Da.shape == means_B_Da.shape == stdevs_B_Da.shape
        return (stdnormal_B_Da * stdevs_B_Da) + means_B_Da

    def _compute_actiondist_entropy(self, actiondist_B_Pa):
        _, stdevs_B_Da = self._extract_actiondist_params(actiondist_B_Pa)
        return self.distribution.entropy(stdevs_B_Da)


class GaussianGRUPolicy(StochasticPolicy, EzPickle):

    def __init__(self, observation_space, action_space, hidden_spec, enable_obsnorm, min_stdev,
                 init_logstdev, state_include_action, varscope_name):
        EzPickle.__init__(self, observation_space, action_space, hidden_spec, enable_obsnorm,
                          min_stdev, init_logstdev, state_include_action, varscope_name)
        self.hidden_spec = hidden_spec
        self.min_stdev = min_stdev
        self.init_logstdev = init_logstdev
        self.state_include_action = state_include_action  # TODO add to stochastic policy
        self._dist = RecurrentGaussian(action_space.shape[0])
        self.prev_actions = None
        self.prev_hiddens = None
        super(GaussianGRUPolicy, self).__init__(observation_space,
                                                action_space,
                                                action_space.shape[0] *
                                                2,  # Mean and diagonal stdev
                                                enable_obsnorm,
                                                varscope_name)

    @property
    def distribution(self):
        return self._dist

    @property
    def recurrent(self):
        return True

    def _make_actiondist_ops(self, obs_B_H_Df):
        B = tf.shape(obs_B_H_Df)[0]
        H = tf.shape(obs_B_H_Df)[1]
        flatobs_B_H_Df = tf.reshape(obs_B_H_Df, tf.pack([B, H, -1]))
        if self.state_include_action:
            net_in = tf.concat(2, [flatobs_B_H_Df, self._prev_actions_B_H_Da])
            net_shape = (np.prod(self.observation_space.shape) + self.action_space.shape[0],)
        else:
            net_in = flatobs_B_H_Df
            net_shape = (np.prod(self.observation_space.shape),)
        with tf.variable_scope('meannet'):
            meannet = nn.GRUNet(net_in, net_shape, self.action_space.shape[0], self.hidden_spec)

        # XXX
        self.hidden_dim = meannet._hidden_dim

        # Action Means
        means_B_H_Da = meannet.output
        # logstdev params
        logstdevs_1_Da = tf.get_variable('logstdevs_1_Da', shape=(1, self.action_space.shape[0]),
                                         initializer=tf.constant_initializer(self.init_logstdev))
        stdevs_1_Da = self.min_stdev + tf.exp(
            logstdevs_1_Da)  # Required for stability of kl computations
        stdevs_B_H_Da = tf.ones_like(means_B_H_Da) * tf.expand_dims(stdevs_1_Da, 1)
        actiondist_B_H_Da = tf.concat(2, [means_B_H_Da, stdevs_B_H_Da])

        steplogstdevs_1_Da = tf.get_variable(
            'steplogstdevs_1_Da', shape=(1, self.action_space.shape[0]),
            initializer=tf.constant_initializer(self.init_logstdev))
        stepstdevs_1_Da = self.min_stdev + tf.exp(steplogstdevs_1_Da)
        stepstdevs_B_Da = tf.ones_like(meannet.step_output) * stepstdevs_1_Da

        if self.state_include_action:
            indim = np.prod(self.observation_space.shape) + self.action_space.shape[0]
        else:
            indim = np.prod(self.observation_space.shape)

        compute_step_mean_std = tfutil.function(
            [meannet.step_input, meannet.step_prev_hidden],
            [meannet.step_output, stepstdevs_B_Da, meannet.step_hidden])

        return actiondist_B_H_Da, meannet.step_input, compute_step_mean_std, meannet.hid_init

    def _extract_actiondist_params(self, actiondist):
        # Aaaaaargh
        if isinstance(actiondist, tf.Tensor):
            ndims = actiondist.get_shape().ndims
        else:
            ndims = actiondist.ndim
        if ndims == 2:
            means = actiondist[:, :self.action_space.shape[0]]
            stdevs = actiondist[:, self.action_space.shape[0]:]
        elif ndims == 3:
            means = actiondist[:, :, :self.action_space.shape[0]]
            stdevs = actiondist[:, :, self.action_space.shape[0]:]
            # means = actiondist[..., :self.action_space.shape[0]]
            # stdevs = actiondist[..., self.action_space.shape[0]:]
        return means, stdevs

    def _make_actiondist_logprobs_ops(self, actiondist_B_H_Pa, input_actions_B_H_Da):
        means_B_H_Da, stdevs_B_H_Da = self._extract_actiondist_params(actiondist_B_H_Pa)
        return self.distribution.log_density_expr(means_B_H_Da, stdevs_B_H_Da, input_actions_B_H_Da)

    def _make_actiondist_kl_ops(self, proposal_actiondist_B_H_Pa, actiondist_B_H_Pa):
        return self.distribution.kl_expr(*map(self._extract_actiondist_params,
                                              [proposal_actiondist_B_H_Pa, actiondist_B_H_Pa]))

    def _compute_actiondist_entropy(self, actiondist_B_H_Pa):
        _, stdevs_B_H_Da = self._extract_actiondist_params(actiondist_B_H_Pa)
        return self.distribution.entropy(stdevs_B_H_Da)

    def reset(self, dones=None):
        if dones is None:
            dones = [True]

        dones = np.asarray(dones)
        if self.prev_actions is None or len(dones) != len(self.prev_actions):
            self.prev_actions = np.zeros((len(dones), self.action_space.shape[0]))
            self.prev_hiddens = np.zeros((len(dones), self.hidden_dim))

        self.prev_actions[dones] = 0.
        self.prev_hiddens[dones] = self._hidden_vec.eval()

    def sample_actions(self, obs_B_Df, deterministic=False):
        B = obs_B_Df.shape[0]
        flat_obs_B_Df = obs_B_Df.reshape((B, -1))
        if self.state_include_action:
            assert self.prev_actions is not None
            net_in_B_Do = np.concatenate([flat_obs_B_Df, self.prev_actions], axis=-1)
        else:
            net_in_B_Do = flat_obs_B_Df

        means_B_Da, stdevs_B_Da, hidden_vec = self.compute_step_actiondist(net_in_B_Do,
                                                                           self.prev_hiddens)
        if deterministic:
            actions_B_Da = means_B_Da
        else:
            actions_B_Da = (np.random.randn(means_B_Da.shape[0], self.action_space.shape[0]) *
                            stdevs_B_Da) + means_B_Da

        prev_actions_B_Da = self.prev_actions
        self.prev_actions = actions_B_Da
        self.prev_hiddens = hidden_vec

        # if self.state_include_action:
        #     actiondist_B_Da = np.concatenate([means_B_Da, stdevs_B_Da, prev_actions_B_Da], axis=1)
        # else:
        actiondist_B_Da = np.concatenate([means_B_Da, stdevs_B_Da], axis=1)
        return actions_B_Da, actiondist_B_Da
