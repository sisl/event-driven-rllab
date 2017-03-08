import argparse
import json
import sys

import numpy as np
import tensorflow as tf

import gym
import rltools.algos.dqn
import rltools.log
import rltools.util
from rltools.samplers.serial import ExperienceReplay
from rltools.policy.eps_greedy import EpsGreedyMLPPolicy


SIMPLE_POLICY_ARCH = '''[
        {"type": "fc", "n": 32},
        {"type": "nonlin", "func": "tanh"},
        {"type": "fc", "n": 32},
        {"type": "nonlin", "func": "tanh"}
    ]
    '''

env = gym.make('CartPole-v0')
#env = gym.make('MountainCar-v0')

policy = EpsGreedyMLPPolicy(env.observation_space, env.action_space,
                                      hidden_spec=SIMPLE_POLICY_ARCH,
                                      enable_obsnorm=True,
                                      tblog='tmp/test_tb', varscope_name='dqn_policy')

target_policy = EpsGreedyMLPPolicy(env.observation_space, env.action_space,
                                      hidden_spec=SIMPLE_POLICY_ARCH,
                                      enable_obsnorm=True,
                                      tblog='tmp/test_tb', varscope_name='target_dqn_policy')

sampler_cls = ExperienceReplay

sampler_args = dict(max_traj_len=500,
                            batch_size=128,
                            min_batch_size=4,
                            max_batch_size=64,
                            batch_rate=40,
                            adaptive=False,
                            initial_exploration=2000,
                            max_experience=5000)

dqn_opt = rltools.algos.dqn.DQNOptimizer(env=env,
                                policy=policy,
                                target_policy=target_policy,
                                discount=0.95,
                                n_iter=1000,
                                n_eval_traj=10,
                                sampler_cls=sampler_cls,
                                sampler_args=sampler_args,
                                start_eps=1.0,
                                end_eps=0.05,
                                clip_grads=None)


args = {'adaptive_batch': False,
         'baseline_type': 'mlp'}
argstr = json.dumps(args, separators=(',',':'), indent=2)
rltools.util.header(argstr)
log_f = rltools.log.TrainingLog('test_log.h5', [('args', argstr)], debug=True)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    dqn_opt.train(sess, log_f, 20)
