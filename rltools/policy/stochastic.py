from __future__ import absolute_import, print_function

from collections import namedtuple
from contextlib import contextmanager

import tensorflow as tf
from gym import spaces

from rltools import nn, optim, tfutil, util
from rltools.policy import Policy


class StochasticPolicy(Policy):

    def __init__(self, observation_space, action_space, num_actiondist_params, enable_obsnorm,
                 varscope_name):
        super(StochasticPolicy, self).__init__(observation_space, action_space)

        with tf.variable_scope(varscope_name) as self.varscope:
            batch_size = None
            if isinstance(self.action_space, spaces.Discrete):
                action_type = tf.int32
                if hasattr(self.action_space, 'ndim'):
                    action_dim = self.action_space.ndim
                else:
                    action_dim = 1
            elif isinstance(self.action_space, spaces.Box):
                action_type = tf.float32
                action_dim = self.action_space.shape[0]
            else:
                raise NotImplementedError()

            if self.recurrent:
                obs_shape = list((batch_size,
                                  None,) + self.observation_space.shape)
                action_shape = [batch_size, None, action_dim]
                actiondist_shape = [batch_size, None, num_actiondist_params]
                advantage_shape = [batch_size, None]
            else:
                obs_shape = list((batch_size,) + self.observation_space.shape)
                action_shape = [batch_size, action_dim]
                actiondist_shape = [batch_size, num_actiondist_params]
                advantage_shape = [batch_size]

            # Action distribution for current policy
            self._obs = tf.placeholder(tf.float32, obs_shape, name='obs')
            with tf.variable_scope('obsnorm'):
                self.obsnorm = (nn.Standardizer if enable_obsnorm else
                                nn.NoOpStandardizer)(self.observation_space.shape)
            self._normalized_obs = self.obsnorm.standardize_expr(self._obs)

            if self.recurrent:
                self._actiondist, self._flatinnet, self.compute_step_actiondist, self._hidden_vec = self._make_actiondist_ops(
                    self._normalized_obs)
            else:
                self._actiondist = self._make_actiondist_ops(self._normalized_obs)

            self._input_action = tf.placeholder(action_type, action_shape,
                                                name='input_actions')  # Action dims FIXME type

            self._logprobs = self._make_actiondist_logprobs_ops(self._actiondist,
                                                                self._input_action)

            # proposal distribution from old policy
            self._proposal_actiondist = tf.placeholder(tf.float32, actiondist_shape,
                                                       name='proposal_actiondist')
            self._proposal_logprobs = self._make_actiondist_logprobs_ops(self._proposal_actiondist,
                                                                         self._input_action)

            # Advantage
            self._advantage = tf.placeholder(tf.float32, advantage_shape, name='advantage')

            if self.recurrent:
                self._valid = tf.placeholder(tf.float32, shape=[None, None], name="valid")
            else:
                self._valid = None

            # Plain pg objective (REINFORCE)
            impweight = tf.exp(self._logprobs - self._proposal_logprobs)
            if self.recurrent:
                self._reinfobj = tf.reduce_sum(impweight * self._advantage *
                                               self._valid) / tf.reduce_sum(self._valid)
            else:
                self._reinfobj = tf.reduce_mean(impweight * self._advantage)  # Surrogate loss

            # KL
            self._kl_coeff = tf.placeholder(tf.float32, name='kl_cost_coeff')
            kl = self._make_actiondist_kl_ops(self._proposal_actiondist, self._actiondist)
            if self.recurrent:
                self._kl = tf.reduce_sum(kl * self._valid) / tf.reduce_sum(self._valid)
            else:
                self._kl = tf.reduce_mean(kl, 0)  # Minimize kl divergence

            # KL Penalty objective for PPO
            self._penobj = self._reinfobj - self._kl_coeff * self._kl

            # All trainable vars done (only _make_* methods)

            # Reading params
            self._param_vars = self.get_variables(trainable=True)
            self._num_params = self.get_num_params(trainable=True)
            self._curr_params_P = tfutil.flatcat(self._param_vars)  # Flatten the params and concat

            self._all_param_vars = self.get_variables()
            self._num_all_params = self.get_num_params()
            self._curr_all_params_PA = tfutil.flatcat(self._all_param_vars)

            # Gradients of objective
            self._reinfobj_grad_P = tfutil.flatcat(
                tfutil.fixedgradients(self._reinfobj, self._param_vars))
            self._penobj_grad_P = tfutil.flatcat(
                tfutil.fixedgradients(self._penobj, self._param_vars))

            # KL gradient for TRPO
            self._kl_grad_P = tfutil.flatcat(tfutil.fixedgradients(self._kl, self._param_vars))

            ins = [self._obs, self._input_action, self._proposal_actiondist, self._advantage]
            if self.recurrent:
                ins.append(self._valid)

            self._compute_internal_normalized_obs = tfutil.function([self._obs],
                                                                    self._normalized_obs)
            self.compute_action_logprobs = tfutil.function([self._obs, self._input_action],
                                                           self._logprobs)
            self.compute_action_dist_params = tfutil.function([self._obs], self._actiondist)

            self.compute_kl_cost = tfutil.function(ins, self._kl)
            self.compute_klgrad = tfutil.function(ins, self._kl_grad_P)
            self.compute_reinfobj_kl = tfutil.function(ins, [self._reinfobj, self._kl])
            self.compute_reinfobj_kl_with_grad = tfutil.function(
                ins, [self._reinfobj, self._kl, self._reinfobj_grad_P])

            self._ngstep = optim.make_ngstep_func(
                self, compute_obj_kl=self.compute_reinfobj_kl,
                compute_obj_kl_with_grad=self.compute_reinfobj_kl_with_grad,
                compute_hvp_helper=self.compute_klgrad)

            # Writing params
            self._flatparams_P = tf.placeholder(tf.float32, [self._num_params], name='flatparams_P')
            # For updating vars directly, e.g. for PPO
            self._assign_params = tfutil.unflatten_into_vars(self._flatparams_P, self._param_vars)

            self._flatallparams_PA = tf.placeholder(tf.float32, [self._num_all_params],
                                                    name='flatallparams_PA')
            self._assign_all_params = tfutil.unflatten_into_vars(self._flatallparams_PA,
                                                                 self._all_param_vars)

            self.set_params = tfutil.function([self._flatparams_P], [], [self._assign_params])
            self.get_params = tfutil.function([], self._curr_params_P)
            self.get_state = tfutil.function([], self._curr_all_params_PA)
            self.set_state = tfutil.function([self._flatallparams_PA], [],
                                             [self._assign_all_params])
            # Treats placeholder self._flatparams_p as gradient for descent
            with tf.variable_scope('optimizer'):
                self._learning_rate = tf.placeholder(tf.float32, name='learning_rate')
                vargrads = tfutil.unflatten_into_tensors(
                    self._flatparams_P, [v.get_shape().as_list() for v in self._param_vars])
                self._take_descent_step = tf.train.AdamOptimizer(
                    learning_rate=self._learning_rate).apply_gradients(
                        util.safezip(vargrads, self._param_vars))

    @property
    def distribution(self):
        raise NotImplementedError()

    def update_obsnorm(self, obs_B_Do, sess):
        """Update norms using moving avg"""
        self.obsnorm.update(obs_B_Do, sess=sess)

    def _make_actiondist_ops(self, obs_B_Df):
        """Ops to compute action distribution parameters

        For Gaussian, these would be mean and std
        For categorical, these would be log probabilities
        """
        raise NotImplementedError()

    def _make_actiondist_logprobs_ops(self, actiondist_B_Pa, input_actions_B_Da):
        raise NotImplementedError()

    def _make_actiondist_kl_ops(self, actiondist_B_Pa, deterministic=False):
        raise NotImplementedError()

    def _sample_from_actiondist(self, actiondist_B_Pa, deterministic=False):
        raise NotImplementedError()

    def _compute_actiondist_entropy(self, actiondist_B_Pa):
        raise NotImplementedError()

    def sample_actions(self, obs_B_Df, deterministic=False, **kwargs):
        """Sample actions conditioned on observations
        (Also returns the params)
        """
        actiondist_B_Pa = self.compute_action_dist_params(obs_B_Df, **kwargs)
        return self._sample_from_actiondist(actiondist_B_Pa, deterministic), actiondist_B_Pa

    def deterministic_action(self, sess, obs_B_Df):
        """Return the argmax of distribution
        """
        actiondist_B_Pa = self.compute_action_dist_params(obs_B_Df, sess=sess)
        a = self._sample_from_actiondist(actiondist_B_Pa, deterministic=True)
        return a[0, 0]

    # TODO penobj computes

    # def set_params(self, params_P, **kwargs):
    #     sess.run(self._assign_params, {self._flatparams_P: params_P})

    # def get_params(self, sess):
    #     params_P = sess.run(self._curr_params_P)
    #     assert params_P.shape == (self._num_params,)
    #     return params_P

    # def get_state(self, sess):
    #     state_PA = sess.run(self._curr_all_params_PA)
    #     assert state_PA.shape == (self._num_all_params,)
    #     return state_PA

    # def set_state(self, sess, state_PA):
    #     sess.run(self._assign_all_params, {self._flatallparams_PA: state_PA})

    @contextmanager
    def try_params(self, params_D, **kwargs):
        orig_params_D = self.get_params(**kwargs)
        self.set_params(params_D, **kwargs)
        yield  # Do what you gotta do
        self.set_params(orig_params_D, **kwargs)

    def take_descent_step(self, sess, grad_P, learning_rate):
        sess.run(self._take_descent_step, {self._flatparams_P: grad_P,
                                           self._learning_rate: learning_rate})
