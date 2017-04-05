import numpy as np
import math
import pdb
import random

# ---- 10 Agents 20 Fires

def fire_param_generator(num_fires_of_each_size):
	normalized_fire_extinguish_times = np.exp(-np.array(list(range(NUM_AGENTS)))) # How long will
	# it take 1, 2, .... agents to extinguish the fire?

	scaling_factor = 3. / normalized_fire_extinguish_times # How much to scale normalized_fire_extinguish_times
		# so that it takes N agents 3 seconds to extinguish this fire

	# set fire rewards quadratically for number of agents
	fire_rewards_of_each_size = [ (x+1)**2 for x in range(NUM_AGENTS) ]

	# compute FIRE_REWARDS and FIRE_PROB_PROFILES for each actual fire
	FIRE_REWARDS = []
	FIRE_PROB_PROFILES = []
	for i, n in enumerate(num_fires_of_each_size):
		FIRE_REWARDS += [fire_rewards_of_each_size[i]] * n
		FIRE_PROB_PROFILES += [ (scaling_factor[i]*normalized_fire_extinguish_times).tolist() ] * n

	return FIRE_REWARDS, FIRE_PROB_PROFILES


NUM_AGENTS = 3
NUM_FIRES = 6

num_fires_of_each_size = [2,2,2]

np.random.seed(100)

GRID_LIM = 1 # Upper and lower bound of grid in x and y dimensions
CT_DISCOUNT_RATE = math.log(0.9)/(-5.) # decay to 90% in 5 seconds
GAMMA = 0. # 2*CT_DISCOUNT_RATE
MAX_SIMTIME = math.log(0.005)/(-CT_DISCOUNT_RATE)  # actions are 0.05% in discounted value

UAV_VELOCITY = 0.03 # m/s
HOLD_TIME = 3. # How long an agent waits when it asks to hold its position

# Generate NUM_FIRES random fire locations

FIRE_LOCATIONS = ( 2.*np.random.random_sample((NUM_FIRES,2)) - 1.).tolist()

normalized_fire_extinguish_times = np.exp(-np.array(list(range(NUM_AGENTS)))) # How long will
	# it take 1, 2, .... agents to extinguish the fire?

scaling_factor = 3. / normalized_fire_extinguish_times # How much to scale normalized_fire_extinguish_times
	# so that it takes N agents 3 seconds to extinguish this fire

# compute number of fires of each size (linear decay with size of fire)


# set fire rewards quadratically for number of agents
fire_rewards_of_each_size = [ (x+1)**2 for x in range(NUM_AGENTS) ]

# compute FIRE_REWARDS and FIRE_PROB_PROFILES for each actual fire
FIRE_REWARDS = []
FIRE_PROB_PROFILES = []
for i, n in enumerate(num_fires_of_each_size):
	FIRE_REWARDS += [fire_rewards_of_each_size[i]] * n
	FIRE_PROB_PROFILES += [ (scaling_factor[i]*normalized_fire_extinguish_times).tolist() ] * n

# Set START_POSITIONS to None so that they are randomized 
START_POSITIONS = None

params = { 'NUM_AGENTS': NUM_AGENTS,
		   'NUM_FIRES': NUM_FIRES,
		   'NUM_FIRES_EACH_SIZE': num_fires_of_each_size,
		   'GRID_LIM': GRID_LIM,
		   'CT_DISCOUNT_RATE': CT_DISCOUNT_RATE,
		   'GAMMA': GAMMA,
		   'MAX_SIMTIME': MAX_SIMTIME,
		   'UAV_VELOCITY': UAV_VELOCITY,
		   'HOLD_TIME': HOLD_TIME,
		   'FIRE_LOCATIONS': FIRE_LOCATIONS,
		   'FIRE_REWARDS': FIRE_REWARDS,
		   'FIRE_PROB_PROFILES': FIRE_PROB_PROFILES,
		   'START_POSITIONS': START_POSITIONS
		     }



## TEST POLICY

import operator 

def distance(arr1,arr2):
	return float(np.linalg.norm( np.array(arr1) - np.array(arr2) ))


def within_epsilon(arr1,arr2):
	return distance(arr1,arr2) < 0.001


## NEW ENV ---
# ---- 10U20F

# NumSamps: 500
# Return: 56.091376214 9.26602384617

# smarter
class test_policy_smart():

	def reset(self, dones = None):
		return

	@property
	def recurrent(self):
		return False

	def get_action(self, obs):

		# Extract meaningful observations
		my_loc = obs[0:2]

		fires = []
		for i in range(5):
			ind_off = 2 + i*6
			loc = obs[ ind_off: ind_off + 2]
			reward, interest, status, secondsleft = tuple(obs[ind_off + 2 : ind_off + 6])
			level = reward
			f = {'loc': loc, 'intr': interest, 'status': status > 0.5, 'lvl': level}
			fires.append(f)

		
		live_fires = [ f for f in fires if f['status']]

		if(len(live_fires) > 0):
			# There are fires alive, so pick the one with biggest gap between interest and level
			interest_gap = [f['lvl'] - f['intr'] for f in live_fires]
			largest_gap_ind = np.argmax(interest_gap)
			fire_to_go_to = live_fires[largest_gap_ind] 

			if(within_epsilon(my_loc, fire_to_go_to['loc'])):
				return 5
			else: 
				return fires.index(fire_to_go_to)


		else:
			# pick a random action
			return random.randint(0,5)

	def get_actions(self, olist):
		return [ self.get_action(o) for o in olist], {}


# smarter
class test_policy_smarter():

	def reset(self, dones = None):
		return

	@property
	def recurrent(self):
		return False

	def get_action(self, obs):


		# Extract meaningful observations
		my_loc = obs[0:2]

		fires = []
		for i in range(5):
			ind_off = 2 + i*5
			dist = obs[ ind_off ]
			reward, interest, status, secondsleft = tuple(obs[ind_off + 1 : ind_off + 5])
			level = reward
			f = {'dist': dist, 'rew': reward, 'intr': interest, 'status': status > 0.5, 'lvl': level, 'secondsleft': secondsleft}
			fires.append(f)

		
		live_fires = [ f for f in fires if f['status']]

		if(len(live_fires) > 0):
			# There are fires alive, so pick the one with biggest gap between interest and level
			expected_time_at_fire = lambda f: float(f['secondsleft']) / (f['intr'] + 1)
			expected_time_to_fire = lambda f: f['dist'] / UAV_VELOCITY

			interest_gap = [float(f['lvl']) / (expected_time_at_fire(f) + expected_time_to_fire(f) )  for f in live_fires]
			largest_gap_ind = np.argmax(interest_gap)
			fire_to_go_to = live_fires[largest_gap_ind] 

			if(fire_to_go_to['dist'] < 1e-2):
				return 5
			else: 
				return fires.index(fire_to_go_to)


		else:
			# pick a random action
			return random.randint(0,5)

	def get_actions(self, olist):
		return [ self.get_action(o) for o in olist], {}


# stupider
class test_policy_stupid():

	def reset(self, dones = None):
		return

	@property
	def recurrent(self):
		return False

	def get_action(self, obs):

		# Extract meaningful observations
		my_loc = obs[0:2]

		fires = []
		for i in range(5):
			ind_off = 2 + i*5
			dist = obs[ ind_off ]
			reward, interest, status, secondsleft = tuple(obs[ind_off + 1 : ind_off + 5])
			level = reward
			f = {'dist': dist, 'rew': reward, 'intr': interest, 'status': status > 0.5, 'lvl': level, 'secondsleft': secondsleft}
			fires.append(f)

		
		live_fires = [ f for f in fires if f['status']]

		if(len(live_fires) > 0):
			# There are fires alive, so pick the one with biggest gap between interest and level
			dists = [f['dist'] for f in live_fires]
			min_dist = np.argmin(dists)
			fire_to_go_to = live_fires[min_dist] 

			if(fire_to_go_to['dist'] < 1e-2):
				return 5
			else: 
				return fires.index(fire_to_go_to)


		else:
			# pick a random action
			return random.randint(0,5)


	def get_actions(self, olist):
		return [ self.get_action(o) for o in olist], {}
