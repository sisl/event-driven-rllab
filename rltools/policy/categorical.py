from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from rltools import nn, tfutil
from rltools.distributions import Categorical, RecurrentCategorical
from rltools.policy.stochastic import StochasticPolicy
from rltools.util import EzPickle


class CategoricalMLPPolicy(StochasticPolicy, EzPickle):

    def __init__(self, observation_space, action_space, hidden_spec, enable_obsnorm, varscope_name):
        EzPickle.__init__(self, observation_space, action_space, hidden_spec, enable_obsnorm,
                          varscope_name)
        self.hidden_spec = hidden_spec
        self._dist = Categorical(action_space.n)
        super(CategoricalMLPPolicy, self).__init__(observation_space, action_space, action_space.n,
                                                   enable_obsnorm, varscope_name)

    @property
    def distribution(self):
        return self._dist

    def _make_actiondist_ops(self, obsfeat_B_Df):
        with tf.variable_scope('hidden'):
            net = nn.FeedforwardNet(obsfeat_B_Df, self.observation_space.shape, self.hidden_spec)
        with tf.variable_scope('out'):
            out_layer = nn.AffineLayer(net.output, net.output_shape, (self.action_space.n,),
                                       Winitializer=tf.zeros_initializer,
                                       binitializer=None)  # TODO action_space

        scores_B_Pa = out_layer.output
        actiondist_B_Pa = scores_B_Pa - tfutil.logsumexp(scores_B_Pa, axis=1)

        return actiondist_B_Pa

    def _make_actiondist_logprobs_ops(self, actiondist_B_Pa, input_actions_B_Da):
        return self.distribution.log_density_expr(actiondist_B_Pa, input_actions_B_Da[:, 0])

    def _make_actiondist_kl_ops(self, proposal_actiondist_B_Pa, actiondist_B_Pa):
        return self.distribution.kl_expr(proposal_actiondist_B_Pa, actiondist_B_Pa)

    def _sample_from_actiondist(self, actiondist_B_Pa, deterministic=False):
        probs_B_A = np.exp(actiondist_B_Pa)
        assert probs_B_A.shape[1] == self.action_space.n
        if deterministic:
            return np.argmax(probs_B_A, axis=1)[:, None]
        return self.distribution.sample(probs_B_A)[:, None]

    def _compute_actiondist_entropy(self, actiondist_B_Pa):
        return self.distribution.entropy(np.exp(actiondist_B_Pa))


class CategoricalGRUPolicy(StochasticPolicy):

    def __init__(self, observation_space, action_space, hidden_spec, enable_obsnorm, varscope_name,
                 state_include_action=True):
        self.hidden_spec = hidden_spec
        self.state_include_action = state_include_action
        self._dist = RecurrentCategorical(action_space.n)
        self.prev_actions = None
        self.prev_hiddens = None

        super(CategoricalGRUPolicy, self).__init__(observation_space, action_space, action_space.n,
                                                   enable_obsnorm, varscope_name)

    @property
    def recurrent(self):
        return True

    @property
    def distribution(self):
        return self._dist

    def _make_actiondist_ops(self, obs_B_H_Df):
        B = tf.shape(obs_B_H_Df)[0]
        H = tf.shape(obs_B_H_Df)[1]
        flatobs_B_H_Df = tf.reshape(obs_B_H_Df, tf.pack([B, H, -1]))
        if self.state_include_action:
            net_in = tf.concat(2, [flatobs_B_H_Df, self._prev_actions_B_H_Da])
            net_shape = (np.prod(self.observation_space.shape) + self.action_space.n,)
        else:
            net_in = flatobs_B_H_Df
            net_shape = (np.prod(self.observation_space.shape),)
        with tf.variable_scope('net'):
            net = nn.GRUNet(net_in, net_shape, self.action_space.n, self.hidden_spec)

        # XXX
        self.hidden_dim = net._hidden_dim

        scores_B_H_Pa = net.output
        actiondist_B_H_Pa = scores_B_H_Pa - tfutil.logsumexp(scores_B_H_Pa, axis=2)

        compute_step_prob = tfutil.function([net.step_input, net.step_prev_hidden],
                                            [net.step_output, net.step_hidden])
        return actiondist_B_H_Pa, net.step_input, compute_step_prob, net.hid_init

    def reset(self, dones=None):
        if dones is None:
            dones = [True]

        dones = np.asarray(dones)
        if self.prev_actions is None or len(dones) != len(self.prev_actions):
            self.prev_actions = np.zeros((len(dones), self.action_space.n))
            self.prev_hiddens = np.zeros((len(dones), self.hidden_dim))

        self.prev_actions[dones] = 0.
        self.prev_hiddens[dones] = self._hidden_vec.eval()

    def _make_actiondist_logprobs_ops(self, actiondist_B_H_Pa, input_actions_B_H_Da):
        return self.distribution.log_density_expr(actiondist_B_H_Pa, input_actions_B_H_Da[:, :, 0])

    def _make_actiondist_kl_ops(self, proposal_actiondist_B_Pa, actiondist_B_Pa):
        return self.distribution.kl_expr(proposal_actiondist_B_Pa, actiondist_B_Pa)

    def _compute_actiondist_entropy(self, actiondist_B_Pa):
        return self.distribution.entropy(np.exp(actiondist_B_Pa))

    def _sample_from_actiondist(self, actiondist_B_Pa, deterministic=False):
        probs_B_A = np.exp(actiondist_B_Pa)
        # XXX
        probs_B_A = probs_B_A / probs_B_A.sum(axis=1)[:, None]
        # XXX
        assert probs_B_A.shape[1] == self.action_space.n
        if deterministic:
            return np.argmax(probs_B_A, axis=1)[:, None]
        return self.distribution.sample(probs_B_A)[:, None]

    def sample_actions(self, obs_B_Df, deterministic=False):
        B = obs_B_Df.shape[0]
        flat_obs_B_Df = obs_B_Df.reshape((B, -1))
        if self.state_include_action:
            assert self.prev_actions is not None
            net_in_B_Do = np.concatenate([flat_obs_B_Df, self.prev_actions], axis=-1)
        else:
            net_in_B_Do = flat_obs_B_Df

        actiondist_B_Pa, hidden_vec = self.compute_step_actiondist(net_in_B_Do, self.prev_hiddens)
        actions_B_Da = self._sample_from_actiondist(actiondist_B_Pa, deterministic)
        prev_actions = self.prev_actions
        self.prev_actions = actions_B_Da
        self.prev_hiddens = hidden_vec

        return actions_B_Da, actiondist_B_Pa
