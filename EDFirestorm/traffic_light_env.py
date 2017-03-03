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


T_INTER = [2,10] # Time range for cars entering system
CAR_INTERSECTION_TIME = 1.0
CT_DISCOUNT_RATE = 0.01



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
		





class TrafficLight(Agent):

	def __init__(self):
		self.resource = simpy.Resource(1)
		self.direction = True # North
		self.light_change_time = 0.
		self.accrued_reward = 0.
		return

	def reset(self):
		self.resource = simpy.Resource(1)
		self.direction = True # North
		self.light_change_time = 0.
		self.accrued_reward = 0.
		return



	@property
	def observation_space(self):
		# Each agent observes: 
			# num cars in its queue for N,S,E,W (2D in [0,max_cars])
			# time until next decision on its neighbors N,S,E,W (4D in [-max_stop_time,max_stop_time])
			#		-ve means traffic is not being allowed in their direction
			# its own sojourn time (prev-action) (1D) 
		max_cars = 20 # cars
		max_stop_time = 100 # seconds
		min_stop_time = 2 # seconds
		return Box( np.array([0]*4 + [-max_stop_time]*4 + [min_stop_time]), 
					np.array([max_cars]*4 + [max_stop_time]*5) )

	@property
	def action_space(self):
		# Actions oscillate between allowing N/S traffic and E/W traffic, 
		#  the action is the amount of time to allow traffic through
		max_stop_time = 100 # seconds
		min_stop_time = 2 # seconds
		return Box( np.array([min_stop_time]), np.array([max_stop_time]))



class TrafficLightEnv(AbstractMAEnv, EzPickle):


	def __init__(self):
		num_agents = 4
		self.env_agents = [SimpleAgent() for _ in range(num_agents)] # NEEDED
		# Internal
		self.n_agents = len(self.env_agents)

		self.max_stop_time = 100 # seconds
		self.min_stop_time = 2 # seconds

		# specificy connectivity as agent j (col) who is [N,S,E,W] of agent i (row)
		self.connectivity = np.array([
			[0,3,0,2], # Agent 1 has 3 to its S and 2 to its W
			[0,4,1,0],
			[1,0,0,4],
			[2,0,3,0]
			])
		# TODO extend connecitivity to handle n_row, n_col arbitrarily

		self.queue = np.array([[0]*4]*self.n_agents ) # initially no one in the queue
		self.allowing_NS = [True]*self.n_agents # True == N/S, False == E/W

		self.times_remaining = [0.]*self.n_agents # amount of time remaining before agent's next action
		self.previous_actions = np.zeros(num_agents, 1) 

		self.accrued_rewards = [0.]*self.n_agents



		EzPickle.__init__(self)

		self.seed()

		self.reset()


	def reset(self):
		self.queue = np.array([[0]*4]*self.n_agents ) # initially no one in the queue
		self.allowing_NS = [True]*self.n_agents # True == N/S, False == E/W

		self.times_remaining = [0.]*self.n_agents

		# Call this with initial actions
		time_range = 10
		return self.step(np.random.rand(self.n_agents, 1) * time_range + self.min_stop_time  )[0]
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

		# Change times remaining, prev_actions given new actions
		for i, a in enumerate(actions.tolist()):
			if a is not None: 
				self.times_remaining[i] = a
				self.prev_actions[i] = a
				# switch agent who just acted's direction
				self.allowing_NS[i] = not self.allowing_NS[i]

		# Compute execution time
		execution_time = min(self.times_remaining)







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