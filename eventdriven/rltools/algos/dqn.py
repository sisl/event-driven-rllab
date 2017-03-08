from __future__ import absolute_import, print_function

import tensorflow as tf
import numpy as np

from rltools import optim, util
from rltools.algos import RLAlgorithm
from rltools.policy.stochastic import StochasticPolicy
from rltools.samplers.serial import ExperienceReplay
from rltools.samplers import evaluate

from copy import deepcopy

class DQNOptimizer(RLAlgorithm):

    def __init__(self, env, policy, target_policy,
                 obsfeat_fn=lambda obs: obs,
                 discount=0.99,
                 n_iter=500,
                 start_iter=0,
                 sample_cls=None, 
                 start_eps=1.0,
                 end_eps=0.05,
                 eval_freq=20,
                 n_eval_traj=50,
                 varscope_name="dqn_opt",
                 opt_learn_rate=0.001,
                 clip_grads=None,
                 sampler_cls=None,
                 sampler_args=dict(max_traj_len=200, batch_size=32, adaptive=False,
                                   min_batch_size=4, max_batch_size=64, batch_rate=40, initial_exploration=5000,
                                   max_experience=10000), **kwargs):

        self.env = env
        self.eval_env = deepcopy(env)
        self.policy = policy
        self.obsfeat_fn = obsfeat_fn
        self.discount = discount
        self.n_iter = n_iter
        self.start_iter = start_iter
        self.eps = start_eps
        self.eps_step = (start_eps - end_eps) / n_iter
        self.eval_freq = eval_freq
        self.n_eval_traj = n_eval_traj
        if sampler_cls is None:
            sampler_cls = ExpereinceReplay
        self.sampler = sampler_cls(self, **sampler_args)
        self.lr = opt_learn_rate
        self.clip_grads = clip_grads

        # define training ops
        self.q_network = policy._actionval_Pa
        self.target_q_network = target_policy._actionval_Pa
        self.observations = policy._obsfeat_B_Df
        self.next_observations = target_policy._obsfeat_B_Df
        with tf.variable_scope(varscope_name):
            self.terminals = tf.placeholder(tf.float32, (None,), name='terminals')
            self.rewards = tf.placeholder(tf.float32, (None,), name='rewards')
            self.action_mask = tf.placeholder(tf.float32, (None, env.action_space.n), name='action_mask')
            self.disc_tf = tf.constant(discount, name='discount')

            self.expected_target_vals = self._expected_val_ops()
            self.td_err_loss = self._td_err_ops()

            self.opt_learn_rate = tf.placeholder(tf.float32, name='learning_rate') # for adaptive learning rate?
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.opt_learn_rate)

            self.grads = self._clip_grads(self.optimizer.compute_gradients(self.td_err_loss), self.clip_grads)
            self.train_op = self.optimizer.apply_gradients(self.grads)

        self.total_time = 0.0

    def train(self, sess, log, save_freq):
        for itr in range(self.start_iter, self.n_iter):
            iter_info = self.step(sess, itr)
            self.eps -= self.eps_step
            self.policy.set_eps(self.eps)
            log.write(iter_info, print_header=itr % 20 == 0)
            if itr % save_freq == 0:
                log.write_snapshot(sess, self.policy, itr)

    def step(self, sess, itr):
        with util.Timer() as t_all:
            # Sample trajs using current policy
            with util.Timer() as t_sample:
                trajbatch, sample_info_fields = self.sampler.sample(sess, itr)

            # Optimize the net
            with util.Timer() as t_step:
               step_print_fields = self.dqn_update(sess, trajbatch, self.lr)

            if itr % self.eval_freq == 0:
                rave = evaluate(self.eval_env, self.obsfeat_fn, lambda ofeat: self.policy.deterministic_action(sess,
                    ofeat), self.sampler.max_traj_len, self.n_eval_traj)
                eval_fileds = [('r_eval', rave, float)]
            else:
                eval_fileds = [('r_eval', None, float)]


        # LOG
        self.total_time += t_all.dt

        fields = [
            ('iter', itr, int)
        ] + sample_info_fields + eval_fileds + [
            ('eps', self.policy.eps, float)
        ] + step_print_fields + [
            ('tsamp', t_sample.dt, float),  # Time for sampling
            ('tstep', t_step.dt, float),
            ('ttotal', self.total_time, float)
        ]
        return fields

    def dqn_update(self, sess, samples, lr):
        # TODO: need action mask
        batch_size = len(samples['observations'])
        feed_terms = np.array(samples['terminals'], dtype=np.float32)
        action_mask = np.zeros((batch_size, self.env.action_space.n))

        for aidx in xrange(self.env.action_space.n):
            action_mask[:,aidx] = samples['actions'] == aidx

        feed = {self.observations: samples['observations'],
                self.rewards: samples['rewards'],
                self.next_observations: samples['next_observations'],
                self.terminals: feed_terms,
                self.action_mask: action_mask,
                self.opt_learn_rate: lr}
        #import IPython
        #IPython.embed()
        _, loss = sess.run([self.train_op, self.td_err_loss], feed_dict=feed)
        return [('loss', loss, float)] 


    def _expected_val_ops(self):
            next_action_vals = tf.stop_gradient(self.target_q_network)
            target_vals = tf.reduce_max(next_action_vals, reduction_indices=[1,]) * self.terminals
            return self.rewards + self.disc_tf * target_vals

    def _td_err_ops(self):
        masked_action_vals = tf.reduce_sum(self.q_network * self.action_mask, reduction_indices=[1,])
        td_err = masked_action_vals - self.expected_target_vals
        return tf.reduce_mean(tf.square(td_err))

    def _clip_grads(self, grads, clip_rate):
        if clip_rate is None: return grads
        for i, (grad, var) in enumerate(grads):
            if grad is not None:
                grads[i] = (tf.clip_by_norm(grad, clip_rate), var)
        return grads
