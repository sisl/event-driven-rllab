import copy
import math
import sys

import numpy as np
import random
#from gym import spaces
from rllab.spaces import Box, Discrete
#from sandbox.rocky.tf.spaces import Box, Discrete


from gym.utils import colorize, seeding

from madrl_environments import AbstractMAEnv, Agent

from rltools.util import EzPickle

from rllab.envs.env_spec import EnvSpec

import pdb

class SimpleAgent(Agent):
	@property
	def observation_space(self):
		return Box(low=-1, high=1, shape=(2,))

	@property
	def action_space(self):
		return Discrete(3)
		#return Box(low=-1, high=1, shape=(2,))



class SimpleEDEnv(AbstractMAEnv, EzPickle):




	def __init__(self):
		self.env_agents = [SimpleAgent() for _ in range(3)] # NEEDED
		# Internal
		self.n_agents = len(self.env_agents)
		self.time_to_event_generator = lambda: np.random.weibull(1.5,1)[0]
		self.time_to_event = np.array([self.time_to_event_generator() for _ in self.env_agents])
		self.sojourn_time = np.array([ 0. for i in self.time_to_event])
		self.global_time = 0.

		EzPickle.__init__(self)

		self.seed()

		self.reset()


	def reset(self):
		self.time_to_event = np.array([self.time_to_event_generator() for _ in self.env_agents])
		self.sojourn_time = np.array([ 0. for i in self.time_to_event])
		self.global_time = 0.

		# Call this with initial actions
		return self.step([0]*self.n_agents)[0]

	def step(self, actions):

		# Convention is:
		#   If an agent is to act on this event, pass an observation and accumulated reward,
		#       otherwise, pass None
		#       "obs" variable will look like: [ [None], [None], [o3_t], [None], [o5_t]  ]
		#       "rewards" will look like:      [  None ,  None ,  r3_r ,  None ,  r5_t   ]
		#   The action returned by the (decentralized) policy will look like
		#                                      [  None ,  None ,  a3_t ,  None ,  a5_t   ]

		# find out whose event triggered
		t_ind = np.argmin(self.time_to_event)
		dt_s = self.time_to_event[t_ind]

		self.global_time += dt_s

		done = self.global_time > 20.

		# increment time elapsed since each agent's last query
		self.sojourn_time += dt_s
		# decerement the time until each agents event next triggers
		self.time_to_event -= dt_s

		# prepare obs, reward for agent(s) who are to be queried
		if not done:
			obs = [ [None] ] * self.n_agents
			obs[t_ind] = [1, self.sojourn_time[t_ind] ]
			rewards = [None] * self.n_agents
			rewards[t_ind] = random.random() # max( [a for a in actions if a != None] )
		else:
			# done so everyone must get a terminal o',r
			obs = [ [1,self.sojourn_time[i]] for i in range(self.n_agents)]
			rewards = [0] * self.n_agents
			rewards[t_ind] = random.random() # max( [a for a in actions if a != None] )

		# reset sojourn time, time till trigger for agent just queried
		self.time_to_event[t_ind] = self.time_to_event_generator()
		self.sojourn_time[t_ind] = 0.


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


if __name__ == "__main__":

	from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
	from sandbox.rocky.tf.envs.base import TfEnv
	from sandbox.rocky.tf.algos.trpo import TRPO
	from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
	from EDFirestorm.EDhelpers import GSMDPBatchSampler, GSMDPCategoricalGRUPolicy, GSMDPGaussianGRUPolicy

	env = SimpleEDEnv()
	env = TfEnv(env)

	policy = CategoricalMLPPolicy(env_spec=env.spec, name = "policy")
	baseline = LinearFeatureBaseline(env_spec=env.spec)
	algo = TRPO(
		env=env,
		policy=policy,
		baseline=baseline,
		discount=0.99999,
		n_itr=75,
		sampler_cls = GSMDPBatchSampler
	)

	algo.train()

	

