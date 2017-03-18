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

from madrl_environments import AbstractMAEnv, Agent

from rltools.util import EzPickle

from rllab.envs.env_spec import EnvSpec

import pdb

import random
from math import exp


T_INTER = [5,6] #[2,15] # Time range for cars entering system
CAR_INTERSECTION_TIME = 1.0
CAR_TRAVEL_TIME = 3.0
CT_DISCOUNT_RATE = math.log(0.9)/(-100.) # decay to 90% in 100 seconds 
MAX_STOP_TIME = 100.
MIN_STOP_TIME = 2.
MAX_SIMTIME = math.log(0.005)/(-CT_DISCOUNT_RATE)  # actions are 0.1% in discounted value
WAIT_REWARD_FACTOR = 0.1 # How much does 1 second of wait cost all traffic lights?
INTERSECTION_CLEARING_TIME = 30.



## --- SIMPY FUNCTIONS


def car_generator(env,traffic_light_list, direction):
	"""Generate new cars that arrive at the gas station."""
	global car_counter
	while(True):
		yield env.timeout(random.randint(*T_INTER))
		env.process(car('Car %d' % car_counter, env, traffic_light_list, direction))
		car_counter+=1

	env.exit()

def car(name, env, traffic_light_list, direction):
	start_time = env.now
	for i, traffic_light in enumerate(traffic_light_list):
		queue_len = len(traffic_light.queues[direction].queue)
		start_time_in_queue = env.now
		with traffic_light.queues[direction].request(priority = 1) as req:
			yield req
			# Take some time to get through intersection
			yield env.timeout(CAR_INTERSECTION_TIME)
		# Give a reward to the traffic light
		traffic_light.accrue_reward(1, env.now)
		# print(name + ' went ' + direction + ' through light %d after waiting %.2f with q-size %d at time %.2f' \
		#  % (traffic_light.name,env.now-start_time_in_queue, queue_len, env.now))

		yield env.timeout(CAR_TRAVEL_TIME)
		
		# Maybe want to send credit to previous stop lights to encourage cooperation?

	end_time = env.now
	# credit all lights passed through equally for wait time
	for traffic_light in traffic_light_list:
		traffic_light.accrue_reward(  -(end_time - start_time)*WAIT_REWARD_FACTOR, end_time )
	env.exit()

def who_triggered(event_list):
	output = [False] * len(event_list)
	for i,e in enumerate(event_list):
		try:
			if(e.ok):
				output[i] = True
		except(AttributeError):
			pass
	return output

def max_simtime_trigger(env, event):
	yield env.timeout(MAX_SIMTIME)
	print('Max simtime reached')
	event.succeed()


## --- ED Env

class TrafficLight(Agent):

	def __init__(self, simpy_env, id_num):
		self.simpy_env = simpy_env
		self.name = id_num
		self.queues = {'north': simpy.PriorityResource(simpy_env,1), 'south': simpy.PriorityResource(simpy_env,1),
			'east': simpy.PriorityResource(simpy_env,1), 'west': simpy.PriorityResource(simpy_env,1)}

		self.direction = (random.random() > 0.5) # True is North
		self.light_change_time = 0.
		self.accrued_reward = 0.
		self.time_trigger = -1
		self.sojourn_time = -1
		return

	def set_neighboring_lights(self,neighbors):
		self.neighbors = neighbors # Expecting array [North Neighbor, S.., E.., West Neighbor]

	def accrue_reward(self, reward, current_time):
		light_change_time = self.light_change_time
		delta_t = current_time - light_change_time
		self.accrued_reward += exp(-delta_t * CT_DISCOUNT_RATE) * reward

	def change_traffic(self, event, time_to_allow):

		self.direction = not self.direction


		if(self.direction): # allowing NS
			with self.queues['east'].request(priority = 0) as req1:
				with self.queues['west'].request(priority = 0) as req2:
					yield req1 and req2
					self.time_trigger = self.simpy_env.now + time_to_allow
					self.sojourn_time = time_to_allow
					yield self.simpy_env.timeout(time_to_allow)
					event.succeed()
					with self.queues['north'].request(priority = 0) as req1:
						with self.queues['south'].request(priority = 0) as req2:
							yield req1 and req2
							yield self.simpy_env.timeout(INTERSECTION_CLEARING_TIME)


		else: # allowing east west
			with self.queues['north'].request(priority = 0) as req1:
				with self.queues['south'].request(priority = 0) as req2:
					yield req1 and req2
					self.time_trigger = self.simpy_env.now + time_to_allow
					self.sojourn_time = time_to_allow
					yield self.simpy_env.timeout(time_to_allow)
					event.succeed()
					with self.queues['east'].request(priority = 0) as req1:
						with self.queues['west'].request(priority = 0) as req2:
							yield req1 and req2
							yield self.simpy_env.timeout(INTERSECTION_CLEARING_TIME)

	def get_obs(self):
		if(self.direction):
			# so that TrafficLight Policy always sees the queue they're going to allow first
			out = [ len(self.queues[d].queue) for d in ['north', 'south', 'east', 'west'] ]
		else:
			out = [ len(self.queues[d].queue) for d in ['east', 'west', 'north', 'south'] ]
		out = out + [ n.time_remaining if n is not None else MAX_STOP_TIME for n in self.neighbors ]
		out = out + [self.sojourn_time]
		return out


	def get_reward(self):
		reward = self.accrued_reward
		self.accrued_reward = 0.
		return reward

	@property
	def time_remaining(self):
		if(self.direction):
			return self.time_trigger - self.simpy_env.now
		else:
			return -self.time_trigger + self.simpy_env.now



	@property
	def observation_space(self):
		# Each agent observes: 
			# num cars in its queue for N,S,E,W (2D in [0,max_cars])
			# time until next decision on its neighbors N,S,E,W (4D in [-max_stop_time,max_stop_time])
			#		-ve means traffic is not being allowed in their direction
			# its own sojourn time (prev-action) (1D) 
		max_cars = 200 # cars
		max_stop_time = MAX_STOP_TIME # seconds
		min_stop_time = MIN_STOP_TIME # seconds
		return Box( np.array([0]*4 + [-max_stop_time]*4 + [min_stop_time]), 
					np.array([max_cars]*4 + [max_stop_time]*5) )

	@property
	def action_space(self):
		# Actions oscillate between allowing N/S traffic and E/W traffic, 
		#  the action is the amount of time to allow traffic through
		max_stop_time = MAX_STOP_TIME # seconds
		min_stop_time = MIN_STOP_TIME # seconds
		return Box( np.array([min_stop_time]), np.array([max_stop_time]))



class TrafficLightEnv(AbstractMAEnv, EzPickle):


	def __init__(self):
		
		self.discount = CT_DISCOUNT_RATE

		num_row_col = 1

		self.n_agents = num_row_col ** 2
		self.max_stop_time = 100 # seconds
		self.min_stop_time = 2 # seconds
		# specify connectivity as East to West across row, North to South across column
		self.connectivity = np.array(list(range(self.n_agents))).reshape((num_row_col,num_row_col))

		# Assigned on reset()
		self.env_agents = [None for _ in range(self.n_agents)] # NEEDED
		self.simpy_env = None
		self.agent_event_list = [None]* self.n_agents


		EzPickle.__init__(self)
		self.seed()
		self.reset()

	def reset(self):

		global car_counter
		car_counter = 0

		self.simpy_env = simpy.Environment()
		env = self.simpy_env
		self.env_agents = [TrafficLight(env, i) for i in range(self.n_agents)]

		self.max_simtime_event = simpy.Event(self.simpy_env)
		self.simpy_env.process( max_simtime_trigger(self.simpy_env, self.max_simtime_event) )

		# Set neighboring lights
		for i in range(self.connectivity.shape[0]):
			for j in range(self.connectivity.shape[1]):
				north = self.env_agents[self.connectivity[i-1,j]] if i > 0 else None
				south = self.env_agents[self.connectivity[i+1,j]] if i < self.connectivity.shape[0]-1 else None
				east = self.env_agents[self.connectivity[i,j-1]] if j > 0 else None
				west = self.env_agents[self.connectivity[i,j+1]] if j < self.connectivity.shape[1]-1 else None
				self.env_agents[self.connectivity[i,j]].set_neighboring_lights([north,south,east,west])

		# Car generators
		for i in range(self.connectivity.shape[0]):
			traffic_light_list = [self.env_agents[j] for j in self.connectivity[i,:].tolist() ]

			# East-bound
			env.process(car_generator(env, traffic_light_list,'east'))
			# West-bound
			env.process(car_generator(env, traffic_light_list[::-1],'west'))

		for i in range(self.connectivity.shape[1]):
			traffic_light_list = [self.env_agents[j] for j in self.connectivity[:,i].tolist() ]

			# South-bound
			env.process(car_generator(env, traffic_light_list,'south'))
			# North-bound
			env.process(car_generator(env, traffic_light_list[::-1],'north'))


		# Call this with initial actions
		time_range = 4
		return self.step( (np.random.rand(self.n_agents, 1) * time_range + self.min_stop_time).tolist()  )[0]

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
				# need to bound action
				action = max( min(a[0], MAX_STOP_TIME), MIN_STOP_TIME )
				event = simpy.Event(self.simpy_env)
				self.agent_event_list[i] = event
				self.simpy_env.process(self.env_agents[i].change_traffic(event, action))

		self.simpy_env.run(until = simpy.AnyOf(self.simpy_env, self.agent_event_list + [self.max_simtime_event]))

		whotriggered = who_triggered(self.agent_event_list)

		# Get next_obs, rewards
		

		# Check if max_simtime_reached
		try:
			self.max_simtime_event.ok
			done = True
			obs = [ e.get_obs() for e in self.env_agents  ]
			rewards = [ e.get_reward() for e in self.env_agents ]
		except(AttributeError):
			done = False
			obs = [ self.env_agents[i].get_obs() if w else [None] for i, w in enumerate(whotriggered)  ]
			rewards = [ self.env_agents[i].get_reward() if w else None for i, w in enumerate(whotriggered)  ]



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

	# TLE = TrafficLightEnv()

	# print('Resetting...')

	# obs = TLE.reset()
	# print('Obs: ', obs)


	# for i in range(3):
	# 	obs, rewards, _, _ = TLE.step( [np.array([2]) if o != [None] else None for o in obs] )
	# 	print('Obs: ', obs)
	# 	print('Reward: ', rewards)

	# quit()


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

	env = TrafficLightEnv()
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
		n_itr=75,
		max_path_length=100000,
		discount=CT_DISCOUNT_RATE,

		# optimizer=ConjugateGradientOptimizer(
  #                                hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)),
		sampler_cls = GSMDPBatchSampler
	)

	algo.train()




