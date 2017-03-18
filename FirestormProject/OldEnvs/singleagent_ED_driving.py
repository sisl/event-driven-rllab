import copy
import math
import sys

import numpy as np
#from gym import spaces
from rllab.spaces import Box, Discrete
# from sandbox.rocky.tf.spaces import Box, Discrete


from gym.utils import colorize, seeding

from madrl_environments import AbstractMAEnv, Agent

from rltools.util import EzPickle

from rllab.envs.env_spec import EnvSpec

import pdb


import random
from math import exp

class SimpleAgent(Agent):
	@property
	def observation_space(self):
		return Box( np.array([-1, -1, 0]), np.array([1000,100, np.inf ]) )

	@property
	def action_space(self):
		return Discrete(7)



class SingleAgentEDDrivingEnv(AbstractMAEnv, EzPickle):

	action_set = [-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0]

	delta_x = 5.0 # show the discrete value of position
	delta_v = 5.0 # show the discrete value of velocity

	speed_camera_location = 350.0
	speed_camera_limit = [25.0,40.0]

	reward_rate_distance = 1.0
	reward_violate_penalty = -200.0

	action_fuel_change = 5.0
	action_fuel_rate = 1.0

	delta_t_min = 0.5  # if delta_t = 0, then set it to 1.0, to avoid keep acting
	delta_t_do_nothing = 2.0 # if car doesn't do anything, the sojourn time for next event = delta_do_nothing




	def __init__(self):
		num_agents = 1
		self.env_agents = [SimpleAgent() for _ in range(num_agents)] # NEEDED
		# Internal
		self.n_agents = len(self.env_agents)


		EzPickle.__init__(self)

		self.seed()

		self.reset()


	def reset(self):
		# current_state = [fuel,position,velocity,last_action]
		self.current_state = [2000.0, 0.0, 52.0, 3]
		# Call this with initial actions
		return self.step(np.array([3]))[0]
		#return self.step(3)[0]

	def step(self, actions):

		# Takes an action set, outputs next observations, accumulated reward, done (boolean), info

		# Convention is:
		#   If an agent is to act on this event, pass an observation and accumulated reward,
		#       otherwise, pass None
		#       "obs" variable will look like: [ [None], [None], [o3_t], [None], [o5_t]  ]
		#       "rewards" will look like:      [  None ,  None ,  r3_r ,  None ,  r5_t   ]
		#   The action returned by the (decentralized) policy will look like
		#                                      [  None ,  None ,  a3_t ,  None ,  a5_t   ]


		current_state = self.current_state
		# pdb.set_trace()
		if(not isinstance( actions, list )):
			actions = [actions]

		next_state, next_obs, next_reward, done = self.transition_event(current_state,actions[0][0])

		self.current_state = next_state

		obs = [next_obs]
		rewards = [next_reward]

		return obs, rewards, done, {'fuel': self.current_state[0], 'done': done }


	def get_reward(self,current_state,next_state):

		# reward is given if the car goes farther
		reward_distance = self.reward_rate_distance * (next_state[1] - current_state[1])

		# penalty is given if the car doesn't pass the speed test at speed camera
		reward_by_penalty = 0

		if current_state[1] < self.speed_camera_location and next_state[1] > self.speed_camera_location:

			average_speed = 0.5 * (current_state[2] + next_state[2])

			if average_speed > self.speed_camera_limit[0] and average_speed < self.speed_camera_limit[1]:

				reward_by_penalty += 0

			else:

				reward_by_penalty += self.reward_violate_penalty

		return reward_distance + reward_by_penalty

	def transition_event(self,current_state,action):
		# returns (next_state, done)

		next_state = [-1.0,-1.0,-1.0,-1]
		previous_action = current_state[3]

		# obtain sojurn time

		if self.action_set[action] == 0.0:

			delta_t = self.delta_t_do_nothing

		elif self.action_set[action] > 0.0:

			delta_v_to_reach = (int(current_state[2]/self.delta_v) + 1)*self.delta_v - current_state[2]

			delta_t = delta_v_to_reach / self.action_set[action]

		elif self.action_set[action] < 0.0:

			delta_v_to_reach = current_state[2] - int(current_state[2]/self.delta_v)*self.delta_v

			delta_t = delta_v_to_reach / abs(self.action_set[action])

		else:

			delta_t = 50.0
			print("sojurn time error")

		# avoid the case that delta_t = 0.0

		if delta_t < 0.01:

			delta_t = self.delta_t_min

		# next_velocity

		next_state[2] = current_state[2] + self.action_set[action] * delta_t

		# next_position

		next_state[1] = (current_state[1] + current_state[2] * delta_t + 0.5 * self.action_set[action] * (delta_t)**2) * random.uniform(1.0,1.2)

		# reward

		r = self.get_reward(current_state,next_state)

		# fuel decrease

		next_state[0] = ( current_state[0] -
										 (1.0 + abs(self.action_set[action])) * self.action_fuel_rate * ( next_state[1] - current_state[1] ) -
										 ((2 * abs(self.action_set[action] - self.action_set[previous_action]))**2) * self.action_fuel_change )

		next_state[3] = action

		next_obs = [ next_state[1], next_state[2], delta_t ]
		next_reward = r

		fuel = next_state[0]
		done = fuel < 0.0
		
		return next_state, next_obs, next_reward, done

	@property
	def spec(self):
		return EnvSpec(
			observation_space=self.env_agents[0].observation_space,
			action_space=self.env_agents[0].action_space,
		)

	@property
	def observation_space(self):
		return self.env_agents[0].observation_space

	@property
	def action_space(self):
		return self.env_agents[0].action_space

	def log_diagnostics(self, paths):
		"""
		Log extra information per iteration based on the collected paths
		"""
		pass

	@property

	@property
	def reward_mech(self):
		return self._reward_mech

	@property
	def agents(self):
		return self.env_agents

	def seed(self, seed=None):
		self.np_random, seed_ = seeding.np_random(seed)
		return [seed_]

	def terminate(self):
		return



if __name__ == "__main__":
	
	from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
	from sandbox.rocky.tf.policies.categorical_gru_policy import CategoricalGRUPolicy
	from sandbox.rocky.tf.core.network import MLP
	from sandbox.rocky.tf.envs.base import TfEnv
	from sandbox.rocky.tf.algos.trpo import TRPO
	from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
	from EDFirestorm.EDhelpers import GSMDPBatchSampler, GSMDPCategoricalGRUPolicy, GSMDPGaussianGRUPolicy
	from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import (ConjugateGradientOptimizer,
                                                                      FiniteDifferenceHvp)
	import tensorflow as tf

	import rllab.misc.logger as logger

	env = SingleAgentEDDrivingEnv()
	env = TfEnv(env)

	# logger.add_tabular_output('./ED_driving_GRU.log')

	feature_network = MLP(name='feature_net', input_shape=(
					env.spec.observation_space.flat_dim + env.spec.action_space.flat_dim,),
										output_dim=7,
										hidden_nonlinearity=tf.nn.tanh,
										hidden_sizes=(32, 32), output_nonlinearity=None)
	feature_network = None

	policy = GSMDPCategoricalGRUPolicy(feature_network = feature_network, env_spec=env.spec, name = "policy")
	# policy = CategoricalMLPPolicy(env_spec=env.spec, name = "policy")
	baseline = LinearFeatureBaseline(env_spec=env.spec)
	algo = TRPO(
		env=env,
		policy=policy,
		baseline=baseline,
		discount=0.,
		n_itr=75,
		optimizer=ConjugateGradientOptimizer(
                                 hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)),
		sampler_cls = GSMDPBatchSampler
	)

	algo.train()

	

