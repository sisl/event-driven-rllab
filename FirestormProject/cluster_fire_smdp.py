## This file alters the game described in simpy_rollout_fire_smdp.py
# Here, agents receive a local observation (location,strength,status,interest) for 5 closest fires
# Also, each fire gets random number of UAV-minutes needed to extinguish it, where the mean is a
#  function of fire level
# Rewards are equal to the fire level
# Fires are clustered close together, but clusters are far apart
# Also new in this version: Agents take epsilon time between get_obs and take_action so that
# two agents supposed to act at the same time dont end up acting sequentially
# Hold time is now normally distributed so that symmetry is broken
# Fires are penalized 1 for trying to extinguish the same fire



import copy
import math
deg = math.pi/180
from math import ceil
import sys
import itertools
import os.path as osp

import numpy as np
from scipy.stats import truncnorm
#from gym import spaces
from rllab.spaces import Box, Discrete
# from sandbox.rocky.tf.spaces import Box, Discrete
import simpy


from gym.utils import colorize, seeding

from eventdriven.madrl_environments import AbstractMAEnv, Agent

from eventdriven.rltools.util import EzPickle

from eventdriven.EDhelpers import SimPyRollout

from rllab.envs.env_spec import EnvSpec

import pdb

import random
from math import exp


## ENVIRONMENT PARAMETERS


GRID_LIM = 1.0 
GAMMA = 0.02 # math.log(0.9)/(-5.)
MAX_SIMTIME = math.log(0.005)/(-GAMMA)

UAV_VELOCITY = 0.015 # m/s
HOLD_TIME = 3. # How long an agent waits when it asks to hold its position
HOLD_TIME_VAR = 0.1*HOLD_TIME

ACTION_WAIT_TIME = 1e-5

UAV_MINS_STD = 0. #1.5
UAV_MINS_AVG = 3.

PRINTING = False
FIRE_DEBUG = False


## --- SIMPY FUNCTIONS

def within_epsilon(arr1,arr2):
	return np.linalg.norm( np.array(arr1) - np.array(arr2) ) < 0.001

def distance(arr1,arr2):
	return float(np.linalg.norm(np.array(arr1) - np.array(arr2)))

## --- ED Env

class UAV(Agent):

	def __init__(self, env, simpy_env, id_num, start_position, goal_position, gamma, policy):
		self.env = env
		self.simpy_env = simpy_env
		self.id_num = id_num
		self.gamma = gamma
		self.policy = policy

		self.observations = []
		self.actions = []
		self.rewards = []
		self.agent_infos = []
		self.env_infos = []
		self.offset_t_sojourn = []

		# Fire Extinguishing specific stuff

		self.start_position = start_position
		self.goal_position = goal_position
		self.action_time = 0.

		self.accrued_reward = 0.

		fire_dists = [ distance(f.location, self.current_position) for f in self.env.fires ]
		closest_five_fires = np.argsort(fire_dists).tolist()[:5]
		self.action_map = closest_five_fires

		self.fire_attacking = -1
		self.fire_interested = -1

		return

	def sim(self):

		obs = self.get_obs()

		while(not self.env.done):

			yield self.simpy_env.timeout(ACTION_WAIT_TIME) # Forces small gap between get_obs and act
			action, agent_info = self.policy(obs)
			self.action_event = self.simpy_env.process(self.take_action(action))
			try:
				yield simpy.AnyOf(self.simpy_env,[self.action_event, self.env.done_event])
			except simpy.Interrupt:
				pass

			reward = self.get_reward()

			self.observations.append(self.env.observation_space.flatten(obs))
			self.actions.append(self.env.action_space.flatten(action))
			self.rewards.append(reward)
			self.agent_infos.append(agent_info)
			self.env_infos.append({})

			obs = self.get_obs()
			self.offset_t_sojourn.append(self.env.observation_space.flatten(obs)[-1])

	def take_action(self, action):

		self.start_position = copy.deepcopy(self.current_position)
		self.action_time = self.simpy_env.now

		# leave any interest party you were in
		if(self.fire_interested != -1):
			self.env.fires[self.fire_interested].leave_interest_party(self)
			self.fire_interested = -1

		hold_current = False
		new_goal = None

		if action >= 5:
			# want to hold
			hold_current = True
			self.goal_position = copy.deepcopy(self.start_position)
			if(PRINTING): print('UAV %d holding at (%.2f, %.2f)' % (self.id_num, self.current_position[0], self.current_position[1]))
			# If we're at a fire, join its extinguish party
			for i, f in enumerate(self.env.fires):
				if within_epsilon(self.current_position, f.location):
					f.join_interest_party(self)
					self.fire_interested = i
					if(self.fire_attacking != i):
						f.join_extinguish_party(self)
						self.fire_attacking = i
					break

			yield self.simpy_env.timeout( self.env.fixed_step( HOLD_TIME + HOLD_TIME_VAR*np.random.normal() ))

		else:
			# assign new goal location, fire interest
			fire_ind = self.action_map[action]
			self.env.fires[fire_ind].join_interest_party(self)
			self.fire_interested = fire_ind
			new_goal = copy.deepcopy(self.env.fires[fire_ind].location)
			# stop attacking any fire you are attacking
			if(self.fire_attacking > -1):
				self.env.fires[self.fire_attacking].leave_extinguish_party(self)
			
			self.goal_position = copy.deepcopy(new_goal)
			travel_time = np.linalg.norm( np.array(self.goal_position) - np.array(self.start_position) ) / UAV_VELOCITY
			if(PRINTING): print('UAV %d is heading from (%.2f, %.2f) to (%.2f, %.2f)' % 
				(self.id_num, self.start_position[0], self.start_position[1], self.goal_position[0], self.goal_position[1] ))

			yield self.simpy_env.timeout(self.env.fixed_step(travel_time))	


	@property
	def time_since_action(self):
		return self.simpy_env.now - self.action_time

	@property
	def current_position(self):
		if( within_epsilon(self.start_position, self.goal_position)):
			return copy.deepcopy(self.start_position)
		# find unit vector in heading direction
		unit_vec = np.array(self.goal_position) - np.array(self.start_position)
		dist_to_travel = np.linalg.norm(unit_vec)
		unit_vec /= dist_to_travel

		# find distance travelled
		distance_travelled = min(self.time_since_action * UAV_VELOCITY,dist_to_travel)

		return ( np.array(self.start_position) + unit_vec * distance_travelled ).tolist()

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
		if(not self.env.done):
			self.accrued_reward += exp(-self.time_since_action * self.gamma) * reward


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
		self.last_update_time = simpy_env.now
		self.interest_party = []
		self.extinguish_event = None

		self.time_until_extinguish = np.inf

		self.level = level
		self.reward = level

		if(UAV_MINS_STD > 0):
			self.uav_seconds_left = float(truncnorm( -UAV_MINS_AVG*level / UAV_MINS_STD, np.inf).rvs(1))
			self.uav_seconds_left = self.uav_seconds_left * UAV_MINS_STD + UAV_MINS_AVG*level
		else:
			self.uav_seconds_left = UAV_MINS_AVG

		if(PRINTING or FIRE_DEBUG):
			print('Fire %d has a %.2f UAV seconds left' % (self.id_num, self.uav_seconds_left))


	def sim(self):
		while(True):
			try:
				self.extinguish_event = self.simpy_env.process(self.try_to_extinguish())
				yield self.extinguish_event
				self.extinguish()
				break
			except simpy.Interrupt:
				continue

	def try_to_extinguish(self):
		yield self.simpy_env.timeout(self.env.fixed_step(self.time_until_extinguish))

	@property
	def uavsecondsleft(self):
		party_size = len(self.extinguish_party)
		now = self.simpy_env.now
		# decrement uav_seconds_left according to how long its been
		# attacked for and by how many agents, since this function
		# was last called
		time_since_last_update = now - self.last_update_time
		decrement = time_since_last_update * party_size
		return self.uav_seconds_left - decrement

		

	def update_extinguish_time(self):

		party_size = len(self.extinguish_party)
		prev_party_size = self.prev_len_extinguish_party
		now = self.simpy_env.now
		# decrement uav_seconds_left according to how long its been
		# attacked for and by how many agents, since this function
		# was last called
		time_since_last_update = now - self.last_update_time
		decrement = time_since_last_update * prev_party_size

		# update state vars
		self.last_update_time = now
		self.prev_len_extinguish_party = party_size
		self.uav_seconds_left -= decrement

		# update event with new time remaining and new party size
		time_to_extinguish = self.uav_seconds_left / party_size if party_size > 0 else np.inf
		self.time_until_extinguish = time_to_extinguish
		try:
			self.extinguish_event.interrupt()
		except RuntimeError:
			pass

		if(FIRE_DEBUG):
			print('Fire %d has extinguish party size %d and %.2f UAV seconds left at time %.2f' %
				 (self.id_num, party_size, self.uav_seconds_left, now))


		return

	def join_interest_party(self, uav):
		if uav not in self.interest_party:
			if(PRINTING): print('UAV %d is joining Fire %d interest party at %.2f' % (uav.id_num, self.id_num, self.simpy_env.now))
			self.interest_party.append(uav)
	def leave_interest_party(self, uav):
		if uav in self.interest_party:
			if(PRINTING): print('UAV %d is leaving Fire %d interest party at %.2f' % (uav.id_num, self.id_num, self.simpy_env.now))
			self.interest_party.remove(uav)

	# Adds an agent to the number of agents trying to extinguish the fire
	def join_extinguish_party(self, uav):
		if(not self.status):
			# Extinguished already
			return self.extinguish_event
		if uav not in self.extinguish_party: 
			if(PRINTING): print('UAV %d is joining Fire %d extinguishing party at %.2f' % (uav.id_num, self.id_num, self.simpy_env.now))
			self.extinguish_party.append(uav)
			if len(self.extinguish_party) > 1:
				# penalize everyone in the part 1
				for uav in self.extinguish_party:
					uav.accrue_reward(-1)
		if(self.status):
			self.update_extinguish_time()

	def leave_extinguish_party(self, uav):
		
		if(not self.status):
			# Extinguished already
			return self.extinguish_event
		if uav in self.extinguish_party: 
			if(PRINTING): print('UAV %d is leaving Fire %d extinguishing party at %.2f' % (uav.id_num, self.id_num, self.simpy_env.now))
			self.extinguish_party.remove(uav)
		if(self.status):
			self.update_extinguish_time()

	def extinguish(self):
		self.status = False
		for a in self.env.env_agents:
			# if(a in self.extinguish_party):
			# 	a.accrue_reward(self.reward)
			# else:
			# 	a.accrue_reward(self.reward)
			a.accrue_reward(self.reward)
		# Interrupt action for all agents in your interest party
		for a in self.interest_party:
			try:
				a.action_event.interrupt()
			except RuntimeError:
				pass

		# set event to one that never triggers
		self.time_until_extinguish = -1
		if(PRINTING or FIRE_DEBUG): print('Fire %d extinguished at %.2f' % (self.id_num, self.simpy_env.now))
		# succeed death event
		self.env.fire_extinguish_events[self.id_num].succeed()
		return






class FireExtinguishingEnv(AbstractMAEnv, EzPickle, SimPyRollout):


	def __init__(self, num_agents, num_fire_clusters, num_fires_per_cluster, gamma,
				 fire_locations = None, start_positions = None, DT = -1):

		EzPickle.__init__(self, num_agents, num_fire_clusters, num_fires_per_cluster, gamma,
				 fire_locations, start_positions, DT)
		
		self.discount = gamma
		self.DT = DT

		self.n_agents = num_agents
		self.n_fires = num_fire_clusters * num_fires_per_cluster
		self.num_fire_clusters = num_fire_clusters
		self.fire_locations = fire_locations
		self.start_positions = start_positions

		# Assigned on reset()
		self.env_agents = [None for _ in range(self.n_agents)] # NEEDED
		self.fires = [None for _ in range(self.n_fires)]
		self.simpy_env = None
		self.uav_events = [] # checks if a UAV needs to act
		self.fire_events = [] # checks if a fire was extinguished
		self.done = False

		self.seed()

	def fixed_step(self, time):
		if(np.isinf(time)):
			return time
		elif(self.DT > 0.):
			now = self.simpy_env.now
			return max(float(ceil((now + time) / self.DT )) * self.DT - now, 0.0)
		else:
			return max(time, 0.0)


	def reset(self):
		# This is a dummy reset just so agent obs/action spaces can be accessed

		self.done = False

		self.simpy_env = simpy.Environment()

		self.fire_extinguish_events = [simpy.Event(self.simpy_env) for i in range(self.n_fires)]

		fire_levels = [1]*self.n_fires

		# we want to randomize
		fire_locations = ( 2.*np.random.random_sample((self.n_fires,2)) - 1.).tolist()
		self.fires = [ Fire(self, self.simpy_env, i, fire_levels[i], fl) 
			for i, fl in enumerate(fire_locations)  ]

		if self.start_positions is not None:
			self.env_agents = [ UAV(self, self.simpy_env, i, sp, sp, self.discount, None) for i,sp in enumerate(self.start_positions) ]
		else:
			# we want to randomize
			start_positions = ( 2.*np.random.random_sample((self.n_agents,2)) - 1.).tolist()
			self.env_agents = [ UAV(self, self.simpy_env, i, sp, sp, self.discount, None) for i,sp in enumerate(start_positions) ]

		return

	def step(self, actions):

		raise NotImplementedError

	def reset_and_sim(self, policies):
		self.simpy_env = simpy.Environment()

		self.done = False

		self.fire_extinguish_events = [simpy.Event(self.simpy_env) for i in range(self.n_fires)]


		if self.fire_locations is True:
			# Use presets
			assert self.num_fire_clusters == 3, 'Only 3 clusters / fires per cluster implemented right now :('
			assert self.n_fires/self.num_fire_clusters == 3, 'Only 3 clusters / fires per cluster implemented right now :('
			R = np.array([[np.cos(120*deg),np.sin(-120*deg)],[np.sin(120*deg), np.cos(120*deg)]])
			f1 = np.reshape(np.array([-0.01, 1]),(2,1))
			f2 = np.reshape(np.array([0.01, 1]),(2,1))
			f3 = np.reshape(np.array([0, 1 - 0.02*math.sin(60*deg)]),(2,1))
			fire_locations = [f1,f2,f3, R.dot(f1),R.dot(f2),R.dot(f3), R.T.dot(f1),R.T.dot(f2),R.T.dot(f3) ]
			fire_locations = [np.reshape(f,(2,)).tolist() for f in fire_locations]
			self.fires = [Fire(self,self.simpy_env, i, 1, fl)
								for i, fl in enumerate(fire_locations)  ]
		else:
			raise NotImplementedError
			# we want to randomize
			fire_locations = ( 2.*np.random.random_sample((self.n_fires,2)) - 1.).tolist()
			self.fires = [ Fire(self, self.simpy_env, i, fire_levels[i], fl) 
				for i, fl in enumerate(fire_locations)  ]

		if self.start_positions is not None:
			self.env_agents = [ UAV(self, self.simpy_env, i, sp, sp, self.discount, policies[i]) for i,sp in enumerate(self.start_positions) ]
		else:
			# we want to randomize
			start_positions = ( 2.*np.random.random_sample((self.n_agents,2)) - 1.).tolist()
			self.env_agents = [ UAV(self, self.simpy_env, i, sp, sp, self.discount, policies[i]) for i,sp in enumerate(start_positions) ]
			
		# Process all UAVs
		agent_events = []
		for uav in self.env_agents:
			agent_events.append(self.simpy_env.process( uav.sim() ))
		# Process all fires
		for fire in self.fires:
			self.simpy_env.process( fire.sim() )

		self.max_simtime_event = self.simpy_env.timeout(MAX_SIMTIME)

		self.done_event = simpy.Event(self.simpy_env)
		self.simpy_env.run(until = simpy.AllOf(self.simpy_env, self.fire_extinguish_events) | self.max_simtime_event )
		self.done_event.succeed()

		self.done = True

		self.simpy_env.run(until = simpy.AllOf(self.simpy_env, agent_events))

		rewards = [uav.get_reward() for uav in self.env_agents]
		if sum(rewards) != 0:
			print('There were unaccounted for rewards')
			[print(r) for r in rewards]
			raise RuntimeError

		# Collect observations, actions, etc.. and return them
		observations = [ u.observations for u in self.env_agents]
		actions = [ u.actions for u in self.env_agents]
		rewards = [ u.rewards for u in self.env_agents]
		agent_infos = [ u.agent_infos for u in self.env_agents]
		env_infos = [ u.env_infos for u in self.env_agents]
		offset_t_sojourn = [ u.offset_t_sojourn for u in self.env_agents ]

		return observations, actions, rewards, agent_infos, env_infos, offset_t_sojourn


	@property
	def spec(self):
		return EnvSpec(
			observation_space=self.env_agents[0].observation_space,
			action_space=self.env_agents[0].action_space,
		)

	@property
	def observation_space(self):

		if self.env_agents[0] is not None:
			return self.env_agents[0].observation_space
		else:
			self.reset()
			return self.env_agents[0].observation_space

	@property
	def action_space(self):
		if self.env_agents[0] is not None:
			return self.env_agents[0].action_space
		else:
			self.reset()
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
	('n_fire_clusters', int, 3, ''),
	('n_fires_per_cluster' , int, 3, ''),
	('fire_locations', list, True, ''),
	('start_positions', list, None, ''),
	('discount', float, GAMMA, ''),
	('GRID_LIM', float, 1.0, ''),
	('MAX_SIMTIME', float, MAX_SIMTIME, ''),
	('UAV_VELOCITY', float, UAV_VELOCITY, ''),
	('HOLD_TIME', float, HOLD_TIME, ''),
	('UAV_MINS_AVG', float, UAV_MINS_AVG, ''),
	('UAV_MINS_STD', float, UAV_MINS_STD, ''),
	('HOLD_TIME_VAR', float, HOLD_TIME_VAR, ''),
	('ACTION_WAIT_TIME', float, ACTION_WAIT_TIME, ''),
	('DT', float, -1., '')
]

from FirestormProject.runners import RunnerParser
from FirestormProject.runners.rurllab import RLLabRunner
import tensorflow as tf


from FirestormProject.test_policy import path_discounted_returns, policy_performance, parallel_policy_performance

if __name__ == "__main__":


	import datetime
	import dateutil

	parser = RunnerParser(ENV_OPTIONS)
	mode = parser._mode
	args = parser.args

	now = datetime.datetime.now(dateutil.tz.tzlocal())
	timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')
	exp_name = 'experiment_%s_dt_%.3f' % (timestamp, args.DT)

	args.exp_name = exp_name
	env =  FireExtinguishingEnv(num_agents = args.n_agents, num_fire_clusters = args.n_fire_clusters, 
							num_fires_per_cluster = args.n_fires_per_cluster, gamma = args.discount,  
			 				fire_locations = args.fire_locations, start_positions = args.start_positions, DT = args.DT)
	
	# run = RLLabRunner(env, args)
	# run()

	# quit()


	filenames = [
	    'experiment_2017_04_22_19_15_17_101782_PDT_dt_-1.000',
	    'experiment_2017_04_22_19_03_39_104449_PDT_dt_0.100',
	    'experiment_2017_04_22_18_51_33_838148_PDT_dt_0.316',
	    'experiment_2017_04_22_18_40_00_951295_PDT_dt_1.000',
	    'experiment_2017_04_22_18_28_44_508570_PDT_dt_3.162',
	    'experiment_2017_04_22_18_17_40_977501_PDT_dt_10.000'
	]

	# experiment_2017_04_22_16_51_28_720596_PDT_dt_1.000
	# 100% (40 of 40) |########################################################################################################################| Elapsed Time: 0:14:29 Time: 0:14:29
	# Mean ADR:  2.74564120508
	# Std ADR: 0.00777526574076
	# experiment_2017_04_21_15_10_08_966990_PDT_dt_-1.000
	# 100% (40 of 40) |########################################################################################################################| Elapsed Time: 0:24:46 Time: 0:24:46
	# Mean ADR:  2.82731574999
	# Std ADR: 0.00812471811702

	# for filename in filenames:
	# 	_, _, adr_list = policy_performance(env = env, gamma = args.discount, num_traj = num_trajs_sim, 
	# 		filename = filename, start_itr = 260, end_itr = 300)

	# num_traj_sim = 100
	# out_dict = {}
	# for filename in filenames:
	# 	out_dict[filename] = parallel_policy_performance(env = env, num_traj = num_traj_sim, 
	# 		filename = filename, start_itr = 260, end_itr = 300)

	# import pickle
	# pickle.dump(out_dict, open('./data/policyperformance.pkl','wb'))

	num_traj_sim = 1
	import glob
	import pickle
	experiments = {-1: './data/*_-1.000', 10: './data/*_10.000', 0.316: './data/*_0.316',
	 3.162: './data/*_3.162', 1: './data/*_1.000', 0.1: './data/*_0.100'  }
	results = {}
	for exp_id, exp_dirs in experiments.items():
		print('Experiment %.2f' % (exp_id))
		filenames = glob.glob(exp_dirs)
		out_dict = {}
		for i, fn in enumerate(filenames):
			out_dict[str(i)] = parallel_policy_performance(env = env, num_traj = num_traj_sim, 
				filename = fn, start_itr = 260, end_itr = 300)
		results[str(exp_id)] = out_dict
		pickle.dump(out_dict, open('./data/ckpt_'+str(exp_id)+'.pkl','wb'))

	pickle.dump(results, open('./data/policyperformance.pkl','wb'))


		








