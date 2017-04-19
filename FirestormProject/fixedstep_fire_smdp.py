## This file alters the game described in new_simpy_fire_smdp.py
# Here, agents receive a local observation (location,strength,status,interest) for 5 closest fires
# Also, each fire gets random number of UAV-minutes needed to extinguish it, where the mean is a
#  function of fire level
# Rewards are equal to the fire level



import copy
import math
import sys
import itertools
import os.path as osp

import numpy as np
from scipy.stats import truncnorm
#from gym import spaces
from rllab.spaces import Box, Discrete
# from sandbox.rocky.tf.spaces import Box, Discrete


from gym.utils import colorize, seeding

from eventdriven.madrl_environments import AbstractMAEnv, Agent

from eventdriven.rltools.util import EzPickle

from rllab.envs.env_spec import EnvSpec

import pdb

import random
from math import exp


## ENVIRONMENT PARAMETERS


GRID_LIM = 1.0 
GAMMA = math.log(0.9)/(-5.)
MAX_SIMTIME = math.log(0.005)/(-GAMMA)

UAV_VELOCITY = 0.015 # m/s
HOLD_TIME = 3. # How long an agent waits when it asks to hold its position

UAV_MINS_STD = 1.5
UAV_MINS_AVG = 3.

DT = 10**(-1) # Time-step

PRINTING = False
FIRE_DEBUG = False



def within_epsilon(arr1,arr2):
	return np.linalg.norm( np.array(arr1) - np.array(arr2) ) < 0.001

def distance(arr1,arr2):
	return float(np.linalg.norm(np.array(arr1) - np.array(arr2)))

## --- ED Env

class UAV(Agent):

	def __init__(self, env, simpy_env, id_num, start_position, goal_position, gamma):
		self.env = env
		self.simpy_env = simpy_env
		self.id_num = id_num
		self.gamma = gamma

		self.current_position = start_position
		self.goal_position = goal_position
		self.hold_clock = 0.
		self.holding = True

		self.accrued_reward = 0.

		fire_dists = [ distance(f.location, self.current_position) for f in self.env.fires ]
		closest_five_fires = np.argsort(fire_dists).tolist()[:5]
		self.action_map = closest_five_fires

		self.fire_attacking = -1
		self.fire_interested = -1

		self.need_new_action = True
		self.time_since_action = 0.

		return


	def get_obs(self):
		obs = copy.deepcopy(self.current_position) # own position
		# find closest fires
		fire_dists = [ distance(f.location, self.current_position) for f in self.env.fires ]
		closest_five_fires = np.argsort(fire_dists).tolist()[:5]
		self.action_map = closest_five_fires
		for f_ind in closest_five_fires:
			f = self.env.fires[f_ind]
			f_obs = [distance(f.location, self.current_position)]
			f_obs += [f.reward, len(f.interest_party)]
			f_obs += [1.] if f.status else [0.]
			f_obs += [f.uavsecondsleft]
			obs += f_obs
		obs += [self.time_since_action]
		return obs


	def get_reward(self):
		reward = self.accrued_reward
		self.accrued_reward = 0.
		return reward

	def accrue_reward(self, reward):
		self.accrued_reward += exp(-self.time_since_action * self.gamma) * reward

	def change_goal(self, hold_current = False, new_goal = None):

		self.need_new_action = False
		self.time_since_action = 0.

		# leave any interest party you were in
		if(self.fire_interested != -1):
			self.env.fires[self.fire_interested].leave_interest_party(self)
			self.fire_interested = -1

		if new_goal is None:
			new_goal = copy.deepcopy(self.goal_position)
		else:
			# assign new goal location, fire interest
			fire_ind = self.action_map[new_goal]
			self.env.fires[fire_ind].join_interest_party(self)
			self.fire_interested = fire_ind
			new_goal = copy.deepcopy(self.env.fires[fire_ind].location)

		if not hold_current:
			# stop attacking any fire you are attacking
			if(self.fire_attacking > -1):
				self.env.fires[self.fire_attacking].leave_extinguish_party(self)
				self.fire_attacking = -1

			self.goal_position = copy.deepcopy(new_goal)
			self.holding = False

		else:
			self.goal_position = copy.deepcopy(self.current_position)

			for i, f in enumerate(self.env.fires):

				if within_epsilon(self.current_position, f.location):
					f.join_interest_party(self)
					self.fire_interested = i
					if(self.fire_attacking != i):
						f.join_extinguish_party(self)
						self.fire_attacking = i
					break

			self.holding = True

		self.hold_clock = HOLD_TIME

	def step(self):

		if(self.holding):
			self.hold_clock = max(0, self.hold_clock - DT)
			if(self.hold_clock == 0):
				self.need_new_action = True
		else:
			dist = distance(self.goal_position, self.current_position)
			if(dist < UAV_VELOCITY * DT):
				self.current_position = copy.deepcopy(self.goal_position)
				self.need_new_action = True

			else:
				# find unit vector in heading direction
				unit_vec = np.array(self.goal_position) - np.array(self.current_position)
				unit_vec /= np.linalg.norm(unit_vec)

				self.current_position = (np.array(self.current_position) + unit_vec * DT*UAV_VELOCITY ).tolist()

		self.time_since_action += DT


	@property
	def observation_space(self):
		# Each agent observes: 
			# Its own x,y coordinates
			# For 5 closest fires: location_x, location_y, strength, interest, status, uavsecondsleft
			# Its sojourn time
		return Box( np.array( [-GRID_LIM] * 2 +  # OWN
							  [0., 0., 0., 0., 0.]*5 + # Fires 
							  [0.] # Sojourn time
							  ), 
					np.array( [GRID_LIM] * 2 +  # OWN
							  [np.inf, 10., np.inf, 1., np.inf]*5 + # Fires 
							  [np.inf] # Sojourn time
							  ),  )

	@property
	def action_space(self):
		# Actions are Fire to go to or STAY
		return Discrete( 5 + # Fires
						 1 ) # stay


class Fire(object):

	def __str__(self):
		return '<{} instance>'.format(type(self).__name__)

	def __init__(self, env, simpy_env, id_num, level, location):
		self.env = env
		self.simpy_env = simpy_env
		self.id_num = id_num
		self.location = location
		self.status = True
		self.extinguish_party = [] # Number of agents trying to extinguish the fire
		self.prev_len_extinguish_party = 0
		self.interest_party = []
		self.time_until_extinguish = np.inf

		self.level = level
		self.reward = level

		self.uav_seconds_left = float(truncnorm( -UAV_MINS_AVG*level / UAV_MINS_STD, np.inf).rvs(1))
		self.uav_seconds_left = self.uav_seconds_left * UAV_MINS_STD + UAV_MINS_AVG*level

		if(PRINTING or FIRE_DEBUG):
			print('Fire %d has a %.2f UAV seconds left' % (self.id_num, self.uav_seconds_left))


	def step(self):
		if(not self.status):
			return

		party_size = len(self.extinguish_party)
		decrement = DT * party_size

		self.uav_seconds_left = max(0, self.uav_seconds_left - decrement)

		if(FIRE_DEBUG and False):
			now = self.env.simtime
			print('Fire %d has extinguish party size %d and %.2f UAV seconds left at time %.2f' %
				 (self.id_num, party_size, self.uav_seconds_left, now))

		if(self.uav_seconds_left == 0):
			self.extinguish()




	@property
	def uavsecondsleft(self):
		return self.uav_seconds_left

	def join_interest_party(self, uav):
		if uav not in self.interest_party:
			if(PRINTING): print('UAV %d is joining Fire %d interest party at %.2f' % (uav.id_num, self.id_num, self.env.simtime))
			self.interest_party.append(uav)
	def leave_interest_party(self, uav):
		if uav in self.interest_party:
			if(PRINTING): print('UAV %d is leaving Fire %d interest party at %.2f' % (uav.id_num, self.id_num, self.env.simtime))
			self.interest_party.remove(uav)

	# Adds an agent to the number of agents trying to extinguish the fire
	def join_extinguish_party(self, uav):
		if(not self.status):
			# Extinguished already
			return None
		if uav not in self.extinguish_party: 
			if(PRINTING): print('UAV %d is joining Fire %d extinguishing party at %.2f' % (uav.id_num, self.id_num, self.env.simtime))
			self.extinguish_party.append(uav)
		# if(PRINTING): print('Fire %d time to extinguish is %.2f' % (self.id_num, self.time_until_extinguish))

	def leave_extinguish_party(self, uav):
		
		if(not self.status):
			# Extinguished already
			return None
		if uav in self.extinguish_party: 
			if(PRINTING): print('UAV %d is leaving Fire %d extinguishing party at %.2f' % (uav.id_num, self.id_num, self.env.simtime))
			self.extinguish_party.remove(uav)
		# if(PRINTING): print('Fire %d time to extinguish is %.2f' % (self.id_num, self.time_until_extinguish))


	def extinguish(self):
		if(not self.status):
			print('Fire was attempting to extinguish more than once')
			return
		self.status = False
		for a in self.env.env_agents:
			# if(a in self.extinguish_party):
			# 	a.accrue_reward(self.reward)
			# else:
			# 	a.accrue_reward(self.reward)
			a.accrue_reward(self.reward)
		for a in self.interest_party:
			a.need_new_action = True

		self.time_until_extinguish = -1
		if(PRINTING or FIRE_DEBUG): print('Fire %d extinguished at %.2f' % (self.id_num, self.env.simtime))
		return



class FixedStepFireExtinguishingEnv(AbstractMAEnv, EzPickle):


	def __init__(self, num_agents, num_fires, num_fires_of_each_size, gamma,
				 fire_locations = None, start_positions = None):

		EzPickle.__init__(self, num_agents, num_fires, num_fires_of_each_size, gamma,
				 fire_locations, start_positions)

		
		self.discount = gamma

		self.n_agents = num_agents
		self.n_fires = num_fires
		self.num_fires_of_each_size = num_fires_of_each_size
		self.fire_locations = fire_locations
		self.start_positions = start_positions

		# Assigned on reset()
		self.env_agents = [None for _ in range(self.n_agents)] # NEEDED
		self.fires = [None for _ in range(self.n_fires)]
		self.simpy_env = None
		self.uav_events = [] # checks if a UAV needs to act
		self.fire_events = [] # checks if a fire was extinguished

		self.simtime = 0.

		self.seed()
		self.reset()

	def reset(self):

		fire_levels = []
		for i, n in enumerate(self.num_fires_of_each_size):
			fire_levels += [i+1] * n

		if self.fire_locations is not None:
			self.fires = [ Fire(self, self.simpy_env, i, fire_levels[i], fl) 
				for i, fl in enumerate(self.fire_locations)  ]
		else:
			# we want to randomize
			fire_locations = ( 2.*np.random.random_sample((self.n_fires,2)) - 1.).tolist()
			self.fires = [ Fire(self, self.simpy_env, i, fire_levels[i], fl) 
				for i, fl in enumerate(fire_locations)  ]

		if self.start_positions is not None:
			self.env_agents = [ UAV(self, self.simpy_env, i, sp, sp, self.discount) for i,sp in enumerate(self.start_positions) ]
		else:
			# we want to randomize
			start_positions = ( 2.*np.random.random_sample((self.n_agents,2)) - 1.).tolist()
			self.env_agents = [ UAV(self, self.simpy_env, i, sp, sp, self.discount) for i,sp in enumerate(start_positions) ]
			


		self.simtime = 0.

		# Step with a hold at start locations
		return self.step( [ 5 ] * self.n_agents  )[0]

	def step(self, actions):

		# Takes an action set, outputs next observations, accumulated reward, done (boolean), info

		# Convention is:
		#   If an agent is to act on this event, pass an observation and accumulated reward,
		#       otherwise, pass None
		#       "obs" variable will look like: [ [None], [None], [o3_t], [None], [o5_t]  ]
		#       "rewards" will look like:      [  None ,  None ,  r3_r ,  None ,  r5_t   ]
		#   The action returned by the (decentralized) policy will look like
		#                                      [  None ,  None ,  a3_t ,  None ,  a5_t   ]

		# prescribe actions to agents
		for i, a in enumerate(actions):
			if a is not None: 
				if a >= 5:
					# Agents wants to hold
					self.env_agents[i].change_goal(hold_current = True)
				else:
					self.env_agents[i].change_goal(new_goal = a)

		# simulate until a new observation is available

		while(True):

			self.simtime += DT

			# -step uavs
			for a in self.env_agents:
				a.step()

			# -step fires
			for f in self.fires:
				f.step()

			# figure out who should act
			agents_to_act = [a.need_new_action for a in self.env_agents]

			# break if any
			if any(agents_to_act):
				break



		done = False
		if(not any([f.status for f in self.fires])):
			done = True

		# Get next_obs, rewards
		if(self.simtime >= MAX_SIMTIME):
			done = True
		

		if(done):
			obs = [ e.get_obs() for e in self.env_agents  ]
			rewards = [ e.get_reward() for e in self.env_agents ]
		else:
			obs = [ self.env_agents[i].get_obs() if w else [None] for i, w in enumerate(agents_to_act)  ]
			rewards = [ self.env_agents[i].get_reward() if w else None for i, w in enumerate(agents_to_act)  ]

		if(PRINTING): print('Obs: ', obs)
		if(PRINTING): print('Reward: ', rewards)

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

	def get_param_values(self):
		return self.__dict__



ENV_OPTIONS = [
	('n_agents', int, 3, ''),
	('n_fires' , int, 6, ''),
	('num_fires_of_each_size', list, [2,2,2], ''),
	('fire_locations', list, None, ''),
	('start_positions', list, None, ''),
	('discount', float, GAMMA, ''),
	('GRID_LIM', float, 1.0, ''),
	('MAX_SIMTIME', float, MAX_SIMTIME, ''),
	('UAV_VELOCITY', float, UAV_VELOCITY, ''),
	('HOLD_TIME', float, HOLD_TIME, ''),
	('UAV_MINS_AVG', float, UAV_MINS_AVG, ''),
	('UAV_MINS_STD', float, UAV_MINS_STD, ''), 
	('DT', float, DT, '')
]

from FirestormProject.runners import RunnerParser
from FirestormProject.runners.rurllab import RLLabRunner

if __name__ == "__main__":

	parser = RunnerParser(ENV_OPTIONS)

	mode = parser._mode
	args = parser.args

	assert args.n_fires >= 5, 'Need 5 or more fires'
	assert args.n_fires == sum(args.num_fires_of_each_size), 'Not exactly as many fires of each size as available fires'

	env =  FixedStepFireExtinguishingEnv(num_agents = args.n_agents, num_fires = args.n_fires, 
								num_fires_of_each_size = args.num_fires_of_each_size, gamma = args.gamma,  
								fire_locations = args.fire_locations, start_positions = args.start_positions)

	from FirestormProject.test_policy import path_discounted_returns

	# print('Fixed-Step %.3f' % (DT))
	# print(path_discounted_returns(env = env, num_traj = 5000, gamma = GAMMA))

	# run = RLLabRunner(env, args)

	# run()

	import tensorflow as tf
	import joblib

	num_trajs_sim = 500

	print('DT: ', DT)

	with tf.Session() as sess:
		from FirestormProject.simpy_rollout_fire_smdp import FireExtinguishingEnv
		obj = joblib.load('./data/experiment_2017_04_10_11_21_38_simpy_rollout/itr_149.pkl')
		policy = obj['policy']
		print('ED Learned Policy')
		print(path_discounted_returns(env = env, num_traj = num_trajs_sim, gamma = GAMMA, policy = policy))

	tf.reset_default_graph()
	with tf.Session() as sess:
		obj = joblib.load('./data/n_parallel1_simpymayhavebugs/experiment_2017_04_07_23_27_23_fixed_10e-1/itr_149.pkl')
		policy = obj['policy']
		print('Fixed-Step 0.1 Learned Policy')
		print(path_discounted_returns(env = env, num_traj = num_trajs_sim, gamma = GAMMA, policy = policy))

	tf.reset_default_graph()
	with tf.Session() as sess:
		obj = joblib.load('./data/n_parallel1_simpymayhavebugs/experiment_2017_04_07_22_13_22_fixed_10e-0p5/itr_149.pkl')
		policy = obj['policy']
		print('Fixed-Step 0.32 Learned Policy')
		print(path_discounted_returns(env = env, num_traj = num_trajs_sim, gamma = GAMMA, policy = policy))

	tf.reset_default_graph()
	with tf.Session() as sess:
		obj = joblib.load('./data/n_parallel1_simpymayhavebugs/experiment_2017_04_07_21_44_55_fixed_10e0/itr_149.pkl')
		policy = obj['policy']
		print('Fixed-Step 1.0 Learned Policy')
		print(path_discounted_returns(env = env, num_traj = num_trajs_sim, gamma = GAMMA, policy = policy))

	tf.reset_default_graph()
	with tf.Session() as sess:
		obj = joblib.load('./data/n_parallel1_simpymayhavebugs/experiment_2017_04_07_20_42_19_fixed_10e0p5/itr_149.pkl')
		policy = obj['policy']
		print('Fixed-Step 3.2 Learned Policy')
		print(path_discounted_returns(env = env, num_traj = num_trajs_sim, gamma = GAMMA, policy = policy))

	tf.reset_default_graph()
	with tf.Session() as sess:
		obj = joblib.load('./data/n_parallel1_simpymayhavebugs/experiment_2017_04_07_18_44_35_fixed_10e1/itr_149.pkl')
		policy = obj['policy']
		print('Fixed-Step 10.0 Learned Policy')
		print(path_discounted_returns(env = env, num_traj = num_trajs_sim, gamma = GAMMA, policy = policy))









