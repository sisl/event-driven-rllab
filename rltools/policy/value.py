from __future__ import absolute_import, print_function

from collections import namedtuple
from contextlib import contextmanager

import tensorflow as tf
from gym import spaces

from rltools import nn, optim, tfutil, util
from rltools.policy import Policy


class ValuePolicy(Policy):

    def __init__(self, obsfeat_space, action_space, num_actiondist_params, enable_obsnorm, tblog,
                 varscope_name):
        super(ValuePolicy, self).__init__(obsfeat_space, action_space)

        with tf.variable_scope(varscope_name) as self.varscope:
            batch_size = None
            assert isinstance(self.action_space,
                              spaces.Discrete), "Value policy support only discrete action spaces"
            action_type = tf.int32
            action_dim = 1
            # Action distribution for current policy
            self._obsfeat_B_Df = tf.placeholder(
                tf.float32, list((batch_size,) + self.obsfeat_space.shape),
                name='obsfeat_B_Df')  # Df = feature dimensions FIXME shape
            with tf.variable_scope('obsnorm'):
                self.obsnorm = (nn.Standardizer if enable_obsnorm else
                                nn.NoOpStandardizer)(self.obsfeat_space.shape[0])
            self._normalized_obsfeat_B_Df = self.obsnorm.standardize_expr(self._obsfeat_B_Df)
            self._actionval_Pa = self._make_actionval_ops(self._normalized_obsfeat_B_Df)
            self._input_action_B_Da = tf.placeholder(
                action_type, [batch_size, action_dim],
                name='input_actions_B_Da')  # Action dims FIXME type

            # All trainable vars done (only _make_* methods)

            # Reading params
            self._param_vars = self.get_variables(trainable=True)
            self._num_params = self.get_num_params(trainable=True)
            self._curr_params_P = tfutil.flatcat(self._param_vars)  # Flatten the params and concat

            # Writing params
            self._flatparams_P = tf.placeholder(tf.float32, [self._num_params], name='flatparams_P')
            # For updating vars directly, e.g. for PPO
            self._assign_params = tfutil.unflatten_into_vars(self._flatparams_P, self._param_vars)

            self._tbwriter = tf.train.SummaryWriter(tblog, graph=tf.get_default_graph())

    def update_obsnorm(self, sess, obs_B_Do):
        """Update norms using moving avg"""
        self.obsnorm.update(sess, obs_B_Do)

    def _make_actionval_ops(self, obsfeat_B_Df):
        """Ops to compute action scores 
        """
        raise NotImplementedError()

    def _compute_internal_normalized_obsfeat(self, sess, obsfeat_B_Df):
        return sess.run(self._normalized_obsfeat_B_Df, {self._obsfeat_B_Df: obsfeat_B_Df})

    def compute_actionvals(self, sess, obsfeat_B_Df):
        """Actually evaluate action distribution params"""
        return sess.run(self._actionval_Pa, {self._obsfeat_B_Df: obsfeat_B_Df})

    def sample_actions(self, sess, obsfeat_B_Df, deterministic=False):
        """Sample actions conditioned on observations
        (Also returns the params)
        """
        actionvals_B_Pa = self.compute_actionvals(sess, obsfeat_B_Df)
        return self._sample_from_actiondist(actionvals_B_Pa, deterministic), None

    def deterministic_action(self, sess, obsfeat_B_Df):
        """Return the argmax of distribution
        """
        actiondist_B_Pa = self.compute_actionvals(sess, obsfeat_B_Df)
        a, _ = self.sample_actions(sess, obsfeat_B_Df, deterministic=True)
        return a[0, 0]

    # TODO penobj computes

    def set_params(self, sess, params_P):
        sess.run(self._assign_params, {self._flatparams_P: params_P})

    def get_params(self, sess):
        params_P = sess.run(self._curr_params_P)
        assert params_P.shape == (self._num_params,)
        return params_P

    @contextmanager
    def try_params(self, sess, params_D):
        orig_params_D = self.get_params(sess)
        self.set_params(sess, params_D)
        yield  # Do what you gotta do
        self.set_params(sess, orig_params_D)
