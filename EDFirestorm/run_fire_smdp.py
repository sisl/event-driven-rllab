import copy
import math
import sys

import numpy as np
#from gym import spaces
from rllab.spaces import Box, Discrete
# from sandbox.rocky.tf.spaces import Box, Discrete
import simpy


from gym.utils import colorize, seeding

from madrl_environments import AbstractMAEnv, Agent

from rltools.util import EzPickle

from rllab.envs.env_spec import EnvSpec

import pdb

import random
from math import exp

from fire_smdp import fire_extinguish



## --- SIMPY FUNCTIONS
def car_generator(env,traffic_light_list):
	"""Generate new cars that arrive at the gas station."""
	for i in itertools.count():
		yield env.timeout(random.randint(*T_INTER))
		env.process(car('Car %d' % i, env, traffic_light_list))

def car(name, env, traffic_light_list):
	for traffic_light in traffic_light_list:
		with traffic_light.resource.request() as req:
			yield req
			# Take some time to get through intersection
			env.timeout(CAR_INTERSECTION_TIME)
		# Give a reward to the traffic light
		current_time = env.now()
		light_change_time = traffic_light.light_change_time
		delta_t = current_time - light_change_time
		self.accrued_reward += exp(-delta_t * CT_DISCOUNT_RATE) * 1.0
		# Maybe want to send credit to previous stop lights to encourage cooperation?
		





class UAV(Agent):

	@property
	def observation_space(self):
		
		return Box( np.array([-1.0] * 7 + [0.]), 
					np.array( [1.0] * 7 + [np.inf]) )

	@property
	def action_space(self):

		return Box( np.array([-1.0, -1.0]), 
					np.array( [1.0, 1.0] ) )



class FirestormSMDPEnv(AbstractMAEnv, EzPickle):


	def __init__(self):
		self.fire_extinguish = fire_extinguish()


		num_agents = self.fire_extinguish.n_uav
		self.env_agents = [UAV() for _ in range(num_agents)] # NEEDED

		

		self.current_state = None
		self.prev_action = None
		


		EzPickle.__init__(self)

		self.seed()

		self.reset()


	def reset(self):

		initial_state = self.fire_extinguish.env_reset()
		self.current_state = initial_state
		initial_action = initial_state[0 : 2*self.fire_extinguish.n_uav]
		self.prev_action = initial_action

		initial_actions = []

		
		for i in range(int( float(len(initial_action))/2)):
			initial_actions.append(np.array(initial_action[2*i:2*i+2]))

		self.rewards_to_give = [0]*len(self.env_agents)


		return self.step( initial_actions )[0]

	def step(self, actions):

		# Takes an action set, outputs next observations, accumulated reward, done (boolean), info

		# Convention is:
		#   If an agent is to act on this event, pass an observation and accumulated reward,
		#       otherwise, pass None
		#       "obs" variable will look like: [ [None], [None], [o3_t], [None], [o5_t]  ]
		#       "rewards" will look like:      [  None ,  None ,  r3_r ,  None ,  r5_t   ]
		#   The action returned by the (decentralized) policy will look like
		#                                      [  None ,  None ,  a3_t ,  None ,  a5_t   ]

		

		curr_actions = self.prev_action
		for i in range(len(actions)):
			if actions[i] is not None:
				curr_actions[2*i] = actions[i][0]
				curr_actions[2*i + 1] = actions[i][1]

		self.prev_action = curr_actions

		next_state, obs, r = self.fire_extinguish.transition(self.current_state,curr_actions)

		self.current_state = next_state

		self.rewards_to_give = [rew + r for rew in self.rewards_to_give]

		fire_status = self.current_state[6:9]
		done = all([f < 0.001 for f in fire_status])

		if not done:
			rewards = [None]*len(obs)
			for ind, o in enumerate(obs):
				if o != [None]:
					rewards[ind] = self.rewards_to_give[ind]
					self.rewards_to_give[ind] = 0
		else:
			rewards = self.rewards_to_give

		



		return obs, rewards, done, {}


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
	
	from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
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

	env = FirestormSMDPEnv()
	env = TfEnv(env)

	# logger.add_tabular_output('./ED_driving_GRU.log')

	# feature_network = MLP(name='feature_net', input_shape=(
	# 				env.spec.observation_space.flat_dim + env.spec.action_space.flat_dim,),
	# 									output_dim=7,
	# 									hidden_nonlinearity=tf.nn.tanh,
	# 									hidden_sizes=(32, 32), output_nonlinearity=None)

	# policy = GSMDPGaussianGRUPolicy(feature_network = feature_network, env_spec=env.spec, name = "policy")
	policy = GaussianMLPPolicy(env_spec=env.spec, name = "policy")
	baseline = LinearFeatureBaseline(env_spec=env.spec)
	algo = TRPO(
		env=env,
		policy=policy,
		baseline=baseline,
		discount=0.,
		n_itr=75,
		max_path_length=1000,
		# optimizer=ConjugateGradientOptimizer(
  #                                hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)),
		sampler_cls = GSMDPBatchSampler
	)

	algo.train()