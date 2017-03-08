import copy
import math
import sys
import itertools

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


## ENVIRONMENT PARAMETERS
# NUM_AGENTS = 2 # Number of agents
# NUM_FIRES = 3 # Number of fires
# GRID_LIM = 1 # Upper and lower bound of grid in x and y dimensions
# CT_DISCOUNT_RATE = math.log(0.9)/(-5.) # decay to 90% in 5 seconds
# MAX_SIMTIME = math.log(0.005)/(-CT_DISCOUNT_RATE)  # actions are 0.05% in discounted value

# UAV_VELOCITY = 0.3 # m/s
# HOLD_TIME = 3. # How long an agent waits when it asks to hold its position

# START_POSITIONS = [[-0.9,-0.9], [0.9,0.9]]
# FIRE_LOCATIONS =  [[-0.5,0.5], [0.5,-0.8], [0.9,-0.9]]

# FIRE_PROB_PROFILES = [ [3.,1.], [10.,3.], [100000., 3.]    ]
# FIRE_REWARDS = [ 1., 2., 5.]


NUM_AGENTS = 4 # Number of agents
NUM_FIRES = 7 # Number of fires
GRID_LIM = 1 # Upper and lower bound of grid in x and y dimensions
CT_DISCOUNT_RATE = math.log(0.9)/(-5.) # decay to 90% in 5 seconds
MAX_SIMTIME = math.log(0.005)/(-CT_DISCOUNT_RATE)  # actions are 0.05% in discounted value

UAV_VELOCITY = 0.3 # m/s
HOLD_TIME = 3. # How long an agent waits when it asks to hold its position

START_POSITIONS = [[-0.9,-0.9], [-0.9,0.9], [0.9,-0.9], [0.9,0.9]]
FIRE_LOCATIONS =  [[-0.8,-0.8], [-0.8,0.8], [0.8,-0.8], [0.8,0.8], 
					[-0.5, 0.], [0., -0.5], [0., 0.]]

FIRE_PROB_PROFILES = [ [1.,0.1, 0.01, 0.001], 
					   [1.,0.1, 0.01, 0.001], 
					   [1.,0.1, 0.01, 0.001],
					   [1.,0.1, 0.01, 0.001],
					   [20.,3., 0.5 , 0.005],
					   [20.,3., 0.5 , 0.005],
					   [1e5, 100., 10., 1. ]    ]
FIRE_REWARDS = [ 1.,1.,1.,1., 5.,5., 20.]

PRINTING = False



## --- SIMPY FUNCTIONS


# Triggers event for when the maximum simulation time has been reached
def max_simtime_trigger(env, event):
	yield env.timeout(MAX_SIMTIME)
	if(PRINTING): print('Max simtime reached')
	event.succeed()

def timeout(env, event, time_out):
	yield env.timeout(time_out)
	event.succeed()

def who_triggered(event_list):
	output = [False] * len(event_list)
	for i,e in enumerate(event_list):
		try:
			if(e.ok):
				output[i] = True
		except(AttributeError):
			pass
	return [i for i, x in enumerate(output) if x]

def within_epsilon(arr1,arr2):
	return np.linalg.norm( np.array(arr1) - np.array(arr2) ) < 0.001

## --- ED Env

class UAV(Agent):

	def __init__(self, env, simpy_env, id_num, start_position, goal_position):
		self.env = env
		self.simpy_env = simpy_env
		self.id_num = id_num

		self.start_position = start_position
		self.goal_position = goal_position
		self.action_time = 0.

		self.accrued_reward = 0.

		self.fire_attacking = -1

		return

	@property
	def time_since_action(self):
		return self.simpy_env.now - self.action_time

	@property
	def current_position(self):
		if( within_epsilon(self.start_position, self.goal_position)):
			return copy.deepcopy(self.start_position)
		# find unit vector in heading direction
		unit_vec = np.array(self.goal_position) - np.array(self.start_position)
		unit_vec /= np.linalg.norm(unit_vec)

		# find distance travelled
		distance_travelled = self.time_since_action * UAV_VELOCITY

		return (np.array(self.start_position) + unit_vec * distance_travelled  ).tolist()

	def get_obs(self):
		obs = copy.deepcopy(self.current_position) # own position
		for i in range(NUM_AGENTS): # everyon else's positions
			if(i != self.id_num):
				obs += self.env.env_agents[i].current_position
		for i in range(NUM_FIRES): # fire statuses
			obs += [1.] if self.env.fires[i].status else [0.]
		obs += [self.time_since_action]
		return obs


	def get_reward(self):
		reward = self.accrued_reward
		self.accrued_reward = 0.
		return reward

	def accrue_reward(self, reward):
		self.accrued_reward += exp(-self.time_since_action * CT_DISCOUNT_RATE) * reward

	def change_goal(self, hold_current = False, new_goal = None):

		if new_goal is None:
			new_goal = copy.deepcopy(self.goal_position)

		event = simpy.Event(self.simpy_env)
		if not hold_current:
			# stop attacking any fire you are attacking
			if(self.fire_attacking > -1):
				self.env.fires[self.fire_attacking].leave_extinguish_party(self)
				self.fire_attacking = -1
			self.start_position = copy.deepcopy(self.current_position)
			self.goal_position = copy.deepcopy(new_goal)
			travel_time = np.linalg.norm( np.array(self.goal_position) - np.array(self.start_position) ) / UAV_VELOCITY
			self.simpy_env.process(timeout(self.simpy_env, event, travel_time))
			if(PRINTING): print('UAV %d is heading from (%.2f, %.2f) to (%.2f, %.2f)' % 
				(self.id_num, self.start_position[0], self.start_position[1], self.goal_position[0], self.goal_position[1] ))
		else:
			# Holding
			self.start_position = copy.deepcopy(self.current_position)
			self.goal_position = copy.deepcopy(self.start_position)
			self.simpy_env.process(timeout(self.simpy_env, event, HOLD_TIME))
			if(PRINTING): print('UAV %d holding at (%.2f, %.2f)' % (self.id_num, self.current_position[0], self.current_position[1]))
			# If we're at a fire, join its extinguish party
			for i, f in enumerate(self.env.fires):
				if within_epsilon(self.current_position, f.location):
					if(self.fire_attacking != i):
						f.join_extinguish_party(self)
						self.fire_attacking = i
					break
		self.env.uav_events[self.id_num] = event
		self.action_time = self.simpy_env.now
		return event


	@property
	def observation_space(self):
		# Each agent observes: 
			# Its own x,y coordinates
			# Every other agents x,y coordinates
			# The fire statuses
			# Its sojourn time
		return Box( np.array( [-GRID_LIM] * 2 +  # OWN
		   					  [-GRID_LIM] * (2*(NUM_AGENTS-1)) + # Other agents
		   					  [0.] * NUM_FIRES + # Fires  
		   					  [0.] # Sojourn time
		   					  ), 
					np.array( [GRID_LIM] * 2 +  # OWN
		   					  [GRID_LIM] * (2*(NUM_AGENTS-1)) + # Other agents
		   					  [1.] * NUM_FIRES + # Fires  
		   					  [np.inf] # Sojourn time
		   					  ),  )

	@property
	def action_space(self):
		# Actions are Fire to go to or STAY
		return Discrete( NUM_FIRES + 1 )


class Fire():

	def __init__(self, env, simpy_env, id_num, reward, prob_profile, location):
		self.env = env
		self.simpy_env = simpy_env
		self.id_num = id_num
		self.reward = reward # on extinguishing
		self.location = location
		assert len(prob_profile) == NUM_AGENTS, 'Number of extinguish probabilities is not the number of agents'
		self.prob_profile = prob_profile # Specifies distribution params for extinguish times for number of agents trying
		self.status = True
		self.extinguish_event = simpy.Event(self.simpy_env) # Gets triggered when the fire is extinguished
		self.extinguish_party = [] # Number of agents trying to extinguish the fire
		self.time_until_extinguish = np.inf

	def update_extinguish_time(self):
		party_size = len(self.extinguish_party)
		if party_size > 0:
			prob_profile_params = self.prob_profile[party_size - 1]
			# Sample a new time
			new_time_until_extinguish = float(np.random.exponential(prob_profile_params))
			if(new_time_until_extinguish < self.time_until_extinguish):
				event = simpy.Event(self.simpy_env)
				self.simpy_env.process(timeout(self.simpy_env, event, new_time_until_extinguish))
				self.extinguish_event = event
				self.time_until_extinguish = new_time_until_extinguish + self.simpy_env.now
		else:
			self.extinguish_event = simpy.Event(self.simpy_env) # will never trigger
			self.time_until_extinguish = np.inf
		# update the event in main env
		self.env.fire_events[self.id_num] = self.extinguish_event
		return

	# Adds an agent to the number of agents trying to extinguish the fire
	def join_extinguish_party(self, uav):
		if(PRINTING): print('UAV %d is joining Fire %d extinguishing party at %.2f' % (uav.id_num, self.id_num, self.simpy_env.now))
		if(not self.status):
			# Extinguished already
			return self.extinguish_event
		self.time_until_extinguish -= self.simpy_env.now
		if uav not in self.extinguish_party: 
			self.extinguish_party.append(uav)
		self.update_extinguish_time()
		if(PRINTING): print('Fire %d time to extinguish is %.2f' % (self.id_num, self.time_until_extinguish - self.simpy_env.now))
		return self.extinguish_event

	def leave_extinguish_party(self, uav):
		if(PRINTING): print('UAV %d is leaving Fire %d extinguishing party at %.2f' % (uav.id_num, self.id_num, self.simpy_env.now))
		if(not self.status):
			# Extinguished already
			return self.extinguish_event
		if uav in self.extinguish_party: 
			self.extinguish_party.remove(uav)
		self.time_until_extinguish = np.inf
		self.update_extinguish_time()
		if(PRINTING): print('Fire %d time to extinguish is %.2f' % (self.id_num, self.time_until_extinguish - self.simpy_env.now))
		return self.extinguish_event

	def extinguish(self):
		self.status = False
		for a in self.env.env_agents:
			a.accrue_reward(self.reward)
		# set event to one that never triggers
		self.extinguish_event = simpy.Event(self.simpy_env)
		self.env.fire_events[self.id_num] = self.extinguish_event
		self.time_until_extinguish = -1
		if(PRINTING): print('Fire %d extinguished at %.2f' % (self.id_num, self.simpy_env.now))
		return






class FireExtinguishingEnv(AbstractMAEnv, EzPickle):


	def __init__(self):
		
		self.discount = CT_DISCOUNT_RATE

		self.n_agents = NUM_AGENTS
		self.n_fires = NUM_FIRES

		# Assigned on reset()
		self.env_agents = [None for _ in range(self.n_agents)] # NEEDED
		self.fires = [None for _ in range(self.n_fires)]
		self.simpy_env = None
		self.uav_events = [] # checks if a UAV needs to act
		self.fire_events = [] # checks if a fire was extinguished


		EzPickle.__init__(self)
		self.seed()
		self.reset()

	def reset(self):

		self.simpy_env = simpy.Environment()
		self.env_agents = [ UAV(self, self.simpy_env, i, sp, sp) for i,sp in enumerate(START_POSITIONS) ]
		self.fires = [ Fire(self, self.simpy_env, i, FIRE_REWARDS[i], FIRE_PROB_PROFILES[i], fl) 
			for i, fl in enumerate(FIRE_LOCATIONS)  ]	

		self.fire_events = [ fire.extinguish_event for fire in self.fires  ]	
		self.uav_events = [simpy.Event(self.simpy_env) for _ in range(self.n_agents)]

		self.max_simtime_event = simpy.Event(self.simpy_env)
		self.simpy_env.process( max_simtime_trigger(self.simpy_env, self.max_simtime_event) )

		# Step with a hold at start locations
		return self.step( [ NUM_FIRES ] * NUM_AGENTS  )[0]

	def step(self, actions):

		# Takes an action set, outputs next observations, accumulated reward, done (boolean), info

		# Convention is:
		#   If an agent is to act on this event, pass an observation and accumulated reward,
		#       otherwise, pass None
		#       "obs" variable will look like: [ [None], [None], [o3_t], [None], [o5_t]  ]
		#       "rewards" will look like:      [  None ,  None ,  r3_r ,  None ,  r5_t   ]
		#   The action returned by the (decentralized) policy will look like
		#                                      [  None ,  None ,  a3_t ,  None ,  a5_t   ]

		for i, a in enumerate(actions):
			if a is not None: 
				if a >= self.n_fires:
					# Agents wants to hold
					self.env_agents[i].change_goal(hold_current = True)
				else:
					self.env_agents[i].change_goal(new_goal = copy.deepcopy(self.fires[a].location))

		self.simpy_env.run(until = simpy.AnyOf(self.simpy_env, self.uav_events + self.fire_events + [self.max_simtime_event]))


		agents_to_act = [False] * self.n_agents
		# check if any fires triggered
		fires_extinguished = who_triggered(self.fire_events)
		for i in fires_extinguished:
			agents_to_act = [True] * self.n_agents
			self.fires[i].extinguish()

		done = False
		if(not any([f.status for f in self.fires])):
			done = True

		# check if any single agent triggered
		uavs_to_act = who_triggered(self.uav_events)
		for i in uavs_to_act:
			agents_to_act[i] = True


		# Get next_obs, rewards
		try:
			# Check if max_simtime_reached
			self.max_simtime_event.ok
			done = True
		except(AttributeError):
			pass
		

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

class test_policy():

	def reset(self, dones = None):
		return

	@property
	def recurrent(self):
		return False
	def get_action_2uav(self, obs):
		my_loc = obs[0:2]		
		fire_statuses = obs[2*NUM_AGENTS : 2*NUM_AGENTS + NUM_FIRES]
		if(fire_statuses[2] > 0.5):
			# Big fire alive
			if(not within_epsilon(my_loc, FIRE_LOCATIONS[2])):
				return 2
			else:
				return 3
		elif(fire_statuses[1] > 0.5):
			# Second biggest alive
			if(not within_epsilon(my_loc, FIRE_LOCATIONS[1])):
				return 1
			else:
				return 3
		else:
			# Smallest alive
			if(not within_epsilon(my_loc, FIRE_LOCATIONS[0])):
				return 0
			else:
				return 3

	def get_action(self, obs):

		my_loc = obs[0:2]		
		fire_statuses = obs[2*NUM_AGENTS : 2*NUM_AGENTS + NUM_FIRES]

		dist = lambda x: np.linalg.norm( np.array(x) - np.array(my_loc))

		closest_small_fire = np.argmin( list(map(dist,FIRE_LOCATIONS[0:4])))

		# Extinguish closest small fire
		if(fire_statuses[closest_small_fire] > 0.5):
			# fire alive
			if(not within_epsilon(my_loc, FIRE_LOCATIONS[closest_small_fire])):
				return closest_small_fire
			else:
				return NUM_FIRES
		# Otherwise extinguish center fire
		elif(fire_statuses[6] > 0.5):
			# fire alive
			if(not within_epsilon(my_loc, FIRE_LOCATIONS[6])):
				return 6
			else:
				return NUM_FIRES
		# Otherwise go to closest medium fire
		else:
			medium_fire_dists = list(map(dist, FIRE_LOCATIONS[4:6]))
			if( np.abs(medium_fire_dists[0] - medium_fire_dists[1]) < 0.1  ):
				closest_medium_fire = 4 if random.random() < 0.5 else 5
			else:
				closest_medium_fire = np.argmin( medium_fire_dists ) + 4
			next_closest_medium_fire = 5 if (closest_medium_fire == 4) else 4
			if(fire_statuses[closest_medium_fire] > 0.5):
				if(not within_epsilon(my_loc, FIRE_LOCATIONS[closest_medium_fire])):
					return closest_medium_fire
				else:
					return NUM_FIRES
			else:
				if(not within_epsilon(my_loc, FIRE_LOCATIONS[next_closest_medium_fire])):
					return next_closest_medium_fire
				else:
					return NUM_FIRES



	def get_actions(self, olist):
		return [ self.get_action(o) for o in olist], {}



if __name__ == "__main__":

	# # Test_Policy
	# env =  FireExtinguishingEnv()

	# from EDFirestorm.EDhelpers import ed_dec_rollout, variable_discount_cumsum
	# agents = test_policy()

	# average_discounted_rewards = []

	# for i in range(100):
	# 	paths = ed_dec_rollout(env, agents)
	# 	for path in paths:
	# 		t_sojourn = path["offset_t_sojourn"]
	# 		discount_gamma = np.exp(-CT_DISCOUNT_RATE*t_sojourn)
	# 		path["returns"] = variable_discount_cumsum(path["rewards"], discount_gamma)
	# 		average_discounted_rewards.append(path["returns"][0])

	# print(len(average_discounted_rewards))
	# print(np.mean(average_discounted_rewards), np.std(average_discounted_rewards))



	# pdb.set_trace()



	

	from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
	from sandbox.rocky.tf.policies.categorical_gru_policy import CategoricalGRUPolicy
	from sandbox.rocky.tf.core.network import MLP
	from sandbox.rocky.tf.envs.base import TfEnv
	from sandbox.rocky.tf.algos.trpo import TRPO
	from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
	from eventdriven.EDhelpers import GSMDPBatchSampler, GSMDPCategoricalGRUPolicy, GSMDPGaussianGRUPolicy
	from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import (ConjugateGradientOptimizer,
                                                                      FiniteDifferenceHvp)
	import tensorflow as tf

	import rllab.misc.logger as logger

	env = FireExtinguishingEnv()
	env = TfEnv(env)

	# logger.add_tabular_output('./FireExtinguishing_MLP_4agent_7fire_150itr_2.log')

	# feature_network = MLP(name='feature_net', input_shape=(
	# 				env.spec.observation_space.flat_dim + env.spec.action_space.flat_dim,),
	# 									output_dim=7,
	# 									hidden_nonlinearity=tf.nn.tanh,
	# 									hidden_sizes=(32, 32), output_nonlinearity=None)

	# policy = GSMDPGaussianGRUPolicy(feature_network = feature_network, env_spec=env.spec, name = "policy")
	policy = CategoricalMLPPolicy(env_spec=env.spec, name = "policy")
	baseline = LinearFeatureBaseline(env_spec=env.spec)
	algo = TRPO(
		env=env,
		policy=policy,
		baseline=baseline,
		n_itr=150,
		max_path_length=100000,
		discount=CT_DISCOUNT_RATE,

		# optimizer=ConjugateGradientOptimizer(
  #                                hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)),
		sampler_cls = GSMDPBatchSampler
	)

	algo.train()




