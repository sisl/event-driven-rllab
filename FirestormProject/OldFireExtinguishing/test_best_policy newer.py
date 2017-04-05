## Test the best trained policy against others



import copy
import math
import sys
import itertools
import os.path as osp

import numpy as np
#from gym import spaces
from rllab.spaces import Box, Discrete
# from sandbox.rocky.tf.spaces import Box, Discrete
import simpy


from gym.utils import colorize, seeding

from eventdriven.madrl_environments import AbstractMAEnv, Agent

from eventdriven.rltools.util import EzPickle

from rllab.envs.env_spec import EnvSpec

import pdb

import random
from math import exp


from newest_simpy_fire_smdp import FireExtinguishingEnv

import tensorflow as tf

import joblib


if __name__ == "__main__":

	from eventdriven.EDhelpers import ed_dec_rollout, variable_discount_cumsum

	with tf.Session() as sess:

		obj = joblib.load('./FirestormProject/FireExtinguishing/Logs/DistObs_WithGamma_3U6F_GRU/itr_100.pkl')

		env = copy.deepcopy(obj['env'])
		policy = obj['policy']

		agents = policy
		print('Learned Policy')

		average_discounted_rewards = []

		GAMMA = 0. #math.log(0.9)/(-5.) # decay to 90% in 5 seconds

		for i in range(200):
			paths = ed_dec_rollout(env, agents)
			for path in paths:
				t_sojourn = path["offset_t_sojourn"]
				discount_gamma = np.exp(-GAMMA*t_sojourn)
				path["returns"] = variable_discount_cumsum(path["rewards"], discount_gamma)
				average_discounted_rewards.append(sum(path["rewards"]))

			if(i%10 == 0):
				print('Iteration: ', i)

		print(len(average_discounted_rewards))
		print(np.mean(average_discounted_rewards), np.std(average_discounted_rewards))

		env = copy.deepcopy(obj['env'])

		from fire_smdp_params import test_policy_smarter
		agents = test_policy_smarter()
		print('Smarter Policy')

		average_discounted_rewards = []

		for i in range(200):
			paths = ed_dec_rollout(env, agents)
			for path in paths:
				t_sojourn = path["offset_t_sojourn"]
				discount_gamma = np.exp(-GAMMA*t_sojourn)
				path["returns"] = variable_discount_cumsum(path["rewards"], discount_gamma)
				average_discounted_rewards.append(sum(path["rewards"]))

			if(i%10 == 0):
				print('Iteration: ', i)

		print(len(average_discounted_rewards))
		print(np.mean(average_discounted_rewards), np.std(average_discounted_rewards))

		env = copy.deepcopy(obj['env'])


		from fire_smdp_params import test_policy_stupid
		agents = test_policy_stupid()
		print('Stupid Policy')

		average_discounted_rewards = []

		for i in range(200):
			paths = ed_dec_rollout(env, agents)
			for path in paths:
				t_sojourn = path["offset_t_sojourn"]
				discount_gamma = np.exp(-GAMMA*t_sojourn)
				path["returns"] = variable_discount_cumsum(path["rewards"], discount_gamma)
				average_discounted_rewards.append(sum(path["rewards"]))

			if(i%10 == 0):
				print('Iteration: ', i)

		print(len(average_discounted_rewards))
		print(np.mean(average_discounted_rewards), np.std(average_discounted_rewards))











