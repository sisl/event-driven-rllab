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
from collections import deque


from gym.utils import colorize, seeding

from eventdriven.madrl_environments import AbstractMAEnv, Agent

from eventdriven.rltools.util import EzPickle

from eventdriven.EDhelpers import SimPyRollout

from rllab.envs.env_spec import EnvSpec

import pdb

import random
from math import exp


## ENVIRONMENT PARAMETERS

ACTION_WAIT_TIME = 1e-5

PRINTING = False
PLOTTING = False




## --- ED Env

class Bus(Agent):

	def __init__(self, env, simpy_env, id_num, gamma, policy, stp_req):
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

		self.accrued_reward = 0.
		self.action_time = 0.

		# Bus specific stuff

		self.next_stop = 0
		self.load = 0
		self.stop_request = stp_req
		self.departure_time = -1 # departure time from current stop (initially unknown)
		self.headways = deque([],2)

		return

	def sim(self):
		yield self.stop_request

		obs = self.get_obs()

		while(not self.env.done):
			if(PRINTING): print('Bus %d obs:' %(self.id_num), obs)

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

		self.action_time = self.simpy_env.now # time of arrival at the next stop
		self.departure_time = -1 # dont know departure time yet

		stop = self.env.stops[self.next_stop]

		passengers_alighting = round(self.load * stop.alight_ratio)
		alighting_time = passengers_alighting * self.env.alighting_time

		headway = self.headways[1] if len(self.headways) > 1 else self.simpy_env.now
		passengers_boarding = round(stop.arrival_rate * headway)
		passengers_boarding = min(passengers_boarding, 
			self.env.bus_capacity - (self.load - passengers_alighting))
		boarding_time = passengers_boarding * self.env.boarding_time

		self.load += passengers_boarding - passengers_alighting

		# wait until all passangers alight and board
		yield self.simpy_env.timeout(max(alighting_time, boarding_time))

		# hold the desired amount of time
		hold_action = action * self.env.hold_time
		yield self.simpy_env.timeout(hold_action)

		if(PRINTING):
			print('Bus %d: P alighted %d, P boarded %d, Load %d, Hold Time %.2f, Cur Time %.2f' % (self.id_num,
				passengers_alighting,passengers_boarding,self.load,hold_action, self.simpy_env.now))

		# leave bus stop
		stop.resource.release(self.stop_request) 

		# depart stop
		self.departure_time = self.simpy_env.now
		stop.departure_times[self.id_num] = self.simpy_env.now

		self.next_stop += 1
		self.next_stop = self.next_stop % self.env.n_stops



		next_stop = self.env.stops[self.next_stop]
		yield self.simpy_env.timeout(next_stop.travel_time)
		self.stop_request = next_stop.resource.request() # wait for all other buses to leave the stop
		yield self.stop_request

		# Arrival
		next_stop.arrival_times[self.id_num] = self.simpy_env.now
		self.headways.append(self.headway)
		if(len(self.headways) > 1):
			# Assign reward to all agents
			H = self.env.planned_headway
			# reward = stop.arrival_rate * \
			# 	( abs(max(self.headways[0],0) - H) - abs(max(self.headways[1],0) - H )  )
			# reward = -next_stop.arrival_rate * abs( self.headways[1] - H) if self.headways[1] > 0 else 0
			reward = -next_stop.arrival_rate * ( self.headways[1] - H)**2 if self.headways[1] > 0 else 0
			for bus in self.env.env_agents:
				bus.accrue_reward(reward)

		if(PRINTING):
			print('Bus %d arrived at stop %d at %.3f' % 
				(self.id_num, next_stop.id_num,self.simpy_env.now))

		return

		



	@property
	def time_since_action(self):
		return self.simpy_env.now - self.action_time

	@property
	def headway(self):
		prev_bus_id_num = ( self.id_num - 1 ) % self.env.n_agents
		next_stop = self.env.stops[self.next_stop]
		prev_departure_time = next_stop.departure_times[prev_bus_id_num]
		self_arrival_time = next_stop.arrival_times[self.id_num]
		if(prev_departure_time >= 0 and self_arrival_time >= 0):
			headway = self_arrival_time - prev_departure_time
		else:
			headway = -1
		return headway

	def get_obs(self):

		obs = [self.next_stop, self.headway, self.load, self.time_since_action]
		
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
		return Box( np.array( [0] + # Number of stops
							  [0] + # Min headway
							  [0] + # Min Load
							  [0.] # Sojourn time
							  ), 
					np.array( [self.env.n_stops] + # Number of stops
							  [np.inf] + # Max Headway
							  [75] + # Max Load
							  [np.inf] # Sojourn time
							  ))

	@property
	def action_space(self):
		# Wait action * T_hold seconds at a stop
		return Discrete( 4 ) 


class Stop(object):

	def __str__(self):
		return '<{} instance>'.format(type(self).__name__)

	def __init__(self, env, simpy_env, id_num, arrival_rate, alight_raticho, travel_time):
		self.env = env
		self.simpy_env = simpy_env
		self.id_num = id_num
		self.arrival_rate = arrival_rate
		self.alight_ratio = alight_ratio
		self.travel_time = travel_time
		self.resource = simpy.Resource(simpy_env, capacity=1)
		self.time_since_last_depart = 0.
		self.departure_times = [-1] * env.n_agents
		self.arrival_times = [-1] * env.n_agents




class BusBunchingEnv(AbstractMAEnv, EzPickle, SimPyRollout):


	def __init__(self, num_agents, num_stops, gamma, kwargs):

		EzPickle.__init__(self, num_agents, num_stops, gamma, kwargs)

		
		self.discount = gamma

		self.n_agents = num_agents
		self.n_stops = num_stops

		self.DT = -1

		# Assigned on reset()
		self.env_agents = [None for _ in range(self.n_agents)] # NEEDED
		self.stops = [None for _ in range(self.n_stops)]
		self.simpy_env = None
		self.bus_events = [] # checks if a UAV needs to act
		self.done = False

		# Env Params
		self.rtk_mode = kwargs.rtk_mode # Expected travel time between stops
		if(self.rtk_mode == 'fixed'):
			self.stop_travel_times = [kwargs.fixed_rtk]*num_stops # 180
		else:
			# E_rtk = Var_rtk = 25
			mu = kwargs.rtk_mu # 3.19927
			sigma = kwargs.stk_sigma # 0.198042
			self.stop_travel_times = np.random.lognormal(mu,sigma,(num_stops)).tolist()
		self.bus_capacity = kwargs.bus_capacity
		self.alighting_time = kwargs.alighting_time
		self.boarding_time = kwargs.boarding_time
		self.planned_headway = kwargs.planned_headway
		self.hold_time = kwargs.hold_time
		self.stop_arrival_rates = kwargs.stop_arrival_rates
		self.stop_alight_ratios = kwargs.stop_alight_ratios
		self.max_simtime = kwargs.max_simtime




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

		self.env_agents = [ Bus(self, self.simpy_env, i, self.discount, None, None) for i in range(self.n_agents) ]

		return

	def step(self, actions):

		raise NotImplementedError

	def reset_and_sim(self, policies):
		self.simpy_env = simpy.Environment()

		self.done = False

		self.stops = [ Stop(env=self, simpy_env=self.simpy_env, id_num=i,
		 arrival_rate=self.stop_arrival_rates[i], alight_ratio=self.stop_alight_ratios[i],
		 travel_time = self.stop_travel_times[i]) for i in range(self.n_stops)]

		self.env_agents = [ Bus(env=self, simpy_env=self.simpy_env, id_num=i,
		 gamma=self.discount, policy=policies[i],
		  stp_req=self.stops[0].resource.request()) for i in range(self.n_agents) ]
			
		# Process all Buses
		agent_events = []
		for bus in self.env_agents:
			agent_events.append(self.simpy_env.process( bus.sim() ))

		self.done_event = simpy.Event(self.simpy_env)
		self.simpy_env.run(until = self.max_simtime )
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

		# Output Info
		if(PLOTTING):
			import matplotlib.pyplot as plt
			outputs = []
			# Plot arrival times
			for i in self.env_agents:
				stops = [ o[0] for o in i.observations ]
				loads = [ o[2] for o in i.observations ]
				travel_times = [ o[3]/60 for o in i.observations ]
				outputs.append( {'agent': i.id_num, 'stops': stops,
				 'loads': loads, 'travel_times': travel_times}  )
				
				plt.plot(np.cumsum(travel_times).tolist(),stops, '*')
			plt.grid()
			plt.show()

			for i in self.env_agents:
				stops = [ o[0] for o in i.observations ]
				loads = [ o[2] for o in i.observations ]
				travel_times = [ o[3]/60 for o in i.observations ]
				
				plt.plot(range(len(loads)), loads, '*-')

			plt.show()

			pdb.set_trace()



		
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



# STOP_ARRIVAL_RATES = [0.75, 1.5, 0.75, 3.0, 1.5, 1.0, 0.75, 0.5, 0.0, 0.0] # Low
STOP_ARRIVAL_RATES = [1.5, 2.25, 1.4, 4.5, 2.55, 1.8, 1.43, 1.05, .75, 0.45] # High
# Convert to seconds
STOP_ARRIVAL_RATES = [s/60 for s in STOP_ARRIVAL_RATES]
STOP_ALIGHT_RATIOS = [1, 0, 0.1, 0.25, 0.25, 0.5, 0.5, 0.1, 0.75, 0.1]

ENV_OPTIONS = [
	('discount', float, 1e-5, ''),
	('max_simtime', float, 3*60*60, ''),
	('n_agents', int, 6, ''),
	('n_stops', int, 10, ''),
	('rtk_mode', str, 'fixed', ''),
	('fixed_rtk', float, 180, ''),
	('bus_capacity', float, 75, ''),
	('alighting_time', float, 1.8, ''),
	('boarding_time', float, 3.0, ''),
	('planned_headway', float, 6*60, ''),
	('hold_time', float, 30, ''),
	('stop_arrival_rates', list, STOP_ARRIVAL_RATES, ''),
	('stop_alight_ratios', list, STOP_ALIGHT_RATIOS, ''),
	('DT', float, -1, '')
]

from BusBunching.runners import RunnerParser
from BusBunching.runners.rurllab import RLLabRunner
import tensorflow as tf

from BusBunching.test_policy import path_discounted_returns
import pickle, joblib

if __name__ == "__main__":

	run = True


	import datetime
	import dateutil

	parser = RunnerParser(ENV_OPTIONS)
	mode = parser._mode
	args = parser.args

	now = datetime.datetime.now(dateutil.tz.tzlocal())
	timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')
	exp_name = 'experiment_%s_dt_%.3f' % (timestamp, args.DT)

	args.exp_name = exp_name

	if(run):

		env =  BusBunchingEnv(num_agents = args.n_agents,
		 num_stops = args.n_stops, gamma = args.discount, kwargs = args)
		
		run = RLLabRunner(env, args)
		run()

	else:
		PLOTTING = True
		args.max_simtime = 6*60*60
		env =  BusBunchingEnv(num_agents = args.n_agents,
		 num_stops = args.n_stops, gamma = args.discount, kwargs = args)
		with tf.Session() as sess:
			obj = joblib.load('./data/experiment_2017_05_29_19_18_07_556228_PDT_dt_-1.000/itr_299.pkl')
			policy = obj['policy']
			path_discounted_returns(env, args.discount, 30, policy=policy, simpy = True, printing = True)
			# path_discounted_returns(env, args.discount, 30, simpy = True, printing = True)


		








