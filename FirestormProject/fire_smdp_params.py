import numpy as np
import math
import pdb
import random

# ---- 2 Agents 3 Fires

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

# ---- 4 Agents 7 Fires

# NUM_AGENTS = 4 # Number of agents
# NUM_FIRES = 7 # Number of fires
# GRID_LIM = 1 # Upper and lower bound of grid in x and y dimensions
# CT_DISCOUNT_RATE = math.log(0.9)/(-5.) # decay to 90% in 5 seconds
# MAX_SIMTIME = math.log(0.005)/(-CT_DISCOUNT_RATE)  # actions are 0.05% in discounted value

# UAV_VELOCITY = 0.3 # m/s
# HOLD_TIME = 3. # How long an agent waits when it asks to hold its position

# START_POSITIONS = [[-0.9,-0.9], [-0.9,0.9], [0.9,-0.9], [0.9,0.9]]
# FIRE_LOCATIONS =  [[-0.8,-0.8], [-0.8,0.8], [0.8,-0.8], [0.8,0.8], 
# 					[-0.5, 0.], [0., -0.5], [0., 0.]]

# FIRE_PROB_PROFILES = [ [1.,0.1, 0.01, 0.001], 
# 					   [1.,0.1, 0.01, 0.001], 
# 					   [1.,0.1, 0.01, 0.001],
# 					   [1.,0.1, 0.01, 0.001],
# 					   [20.,3., 0.5 , 0.005],
# 					   [20.,3., 0.5 , 0.005],
# 					   [1e5, 100., 10., 1. ]    ]
# FIRE_REWARDS = [ 1.,1.,1.,1., 5.,5., 20.]

# ---- 30 Agents 50 Fires

# NUM_AGENTS = 30
# NUM_FIRES = 50

# np.random.seed(10)

# GRID_LIM = 1 # Upper and lower bound of grid in x and y dimensions
# CT_DISCOUNT_RATE = math.log(0.9)/(-5.) # decay to 90% in 5 seconds
# MAX_SIMTIME = math.log(0.005)/(-CT_DISCOUNT_RATE)  # actions are 0.05% in discounted value

# UAV_VELOCITY = 0.03 # m/s
# HOLD_TIME = 3. # How long an agent waits when it asks to hold its position

# # Generate NUM_FIRES random fire locations

# FIRE_LOCATIONS = ( 2.*np.random.random_sample((NUM_FIRES,2)) - 1.).tolist()

# normalized_fire_extinguish_times = np.exp(-np.array(list(range(NUM_AGENTS)))) # How long will
# 	# it take 1, 2, .... agents to extinguish the fire?

# scaling_factor = 3. / normalized_fire_extinguish_times # How much to scale normalized_fire_extinguish_times
# 	# so that it takes N agents 3 seconds to extinguish this fire


# # compute number of fires of each size (linear decay with size of fire)
# sf = float(2*NUM_FIRES)/(NUM_AGENTS**2 + NUM_AGENTS)
# num_fires_of_each_size = [ int((x + 1) * sf)  for x in range(NUM_AGENTS) ][::-1]
# # ensure they sum to num_fires
# num_fires_of_each_size[0] = NUM_FIRES - sum(num_fires_of_each_size[1:])

# # set fire rewards quadratically for number of agents
# fire_rewards_of_each_size = [ (x+1)**2 for x in range(NUM_AGENTS) ]

# # compute FIRE_REWARDS and FIRE_PROB_PROFILES for each actual fire
# FIRE_REWARDS = []
# FIRE_PROB_PROFILES = []
# for i, n in enumerate(num_fires_of_each_size):
# 	FIRE_REWARDS += [fire_rewards_of_each_size[i]] * n
# 	FIRE_PROB_PROFILES += [ (scaling_factor[i]*normalized_fire_extinguish_times).tolist() ] * n

# # Set START_POSITIONS to None so that they are randomized 
# START_POSITIONS = None

# ---- 10 Agents 20 Fires

NUM_AGENTS = 10
NUM_FIRES = 20

np.random.seed(100)

GRID_LIM = 1 # Upper and lower bound of grid in x and y dimensions
CT_DISCOUNT_RATE = math.log(0.9)/(-5.) # decay to 90% in 5 seconds
MAX_SIMTIME = math.log(0.005)/(-CT_DISCOUNT_RATE)  # actions are 0.05% in discounted value

UAV_VELOCITY = 0.3 # m/s
HOLD_TIME = 3. # How long an agent waits when it asks to hold its position

# Generate NUM_FIRES random fire locations

FIRE_LOCATIONS = ( 2.*np.random.random_sample((NUM_FIRES,2)) - 1.).tolist()

normalized_fire_extinguish_times = np.exp(-np.array(list(range(NUM_AGENTS)))) # How long will
	# it take 1, 2, .... agents to extinguish the fire?

scaling_factor = 3. / normalized_fire_extinguish_times # How much to scale normalized_fire_extinguish_times
	# so that it takes N agents 3 seconds to extinguish this fire


# compute number of fires of each size (linear decay with size of fire)
sf = float(2*NUM_FIRES)/(NUM_AGENTS**2 + NUM_AGENTS)
num_fires_of_each_size = [8, 6, 4, 2]

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
		   'GRID_LIM': GRID_LIM,
		   'CT_DISCOUNT_RATE': CT_DISCOUNT_RATE,
		   'MAX_SIMTIME': MAX_SIMTIME,
		   'UAV_VELOCITY': UAV_VELOCITY,
		   'HOLD_TIME': HOLD_TIME,
		   'FIRE_LOCATIONS': FIRE_LOCATIONS,
		   'FIRE_REWARDS': FIRE_REWARDS,
		   'FIRE_PROB_PROFILES': FIRE_PROB_PROFILES,
		   'START_POSITIONS': START_POSITIONS
		     }


# pdb.set_trace()


## TEST POLICY

import operator 

def distance(arr1,arr2):
	return float(np.linalg.norm( np.array(arr1) - np.array(arr2) ))


def within_epsilon(arr1,arr2):
	return distance(arr1,arr2) < 0.001

# ---- 2 Agents 3 Fires

# class test_policy():

# 	def reset(self, dones = None):
# 		return

# 	@property
# 	def recurrent(self):
# 		return False
# 	def get_action(self, obs):
# 		my_loc = obs[0:2]		
# 		fire_statuses = obs[2*NUM_AGENTS : 2*NUM_AGENTS + NUM_FIRES]
# 		if(fire_statuses[2] > 0.5):
# 			# Big fire alive
# 			if(not within_epsilon(my_loc, FIRE_LOCATIONS[2])):
# 				return 2
# 			else:
# 				return 3
# 		elif(fire_statuses[1] > 0.5):
# 			# Second biggest alive
# 			if(not within_epsilon(my_loc, FIRE_LOCATIONS[1])):
# 				return 1
# 			else:
# 				return 3
# 		else:
# 			# Smallest alive
# 			if(not within_epsilon(my_loc, FIRE_LOCATIONS[0])):
# 				return 0
# 			else:
# 				return 3

# 	def get_actions(self, olist):
# 		return [ self.get_action(o) for o in olist], {}


# import operator
# class test_policy():

# 	def reset(self, dones = None):
# 		return

# 	@property
# 	def recurrent(self):
# 		return False

# 	def get_action(self, obs):

# 		my_loc = obs[0:2]		
# 		fire_statuses = obs[2*NUM_AGENTS : 2*NUM_AGENTS + NUM_FIRES]

# 		dist = lambda x: np.linalg.norm( np.array(x) - np.array(my_loc))

# 		closest_small_fire = np.argmin( list(map(dist,FIRE_LOCATIONS[0:4])))

# 		# Extinguish closest small fire
# 		if(fire_statuses[closest_small_fire] > 0.5):
# 			# fire alive
# 			if(not within_epsilon(my_loc, FIRE_LOCATIONS[closest_small_fire])):
# 				return closest_small_fire
# 			else:
# 				return NUM_FIRES
# 		# Otherwise extinguish center fire
# 		elif(fire_statuses[6] > 0.5):
# 			# fire alive
# 			if(not within_epsilon(my_loc, FIRE_LOCATIONS[6])):
# 				return 6
# 			else:
# 				return NUM_FIRES
# 		# Otherwise go to closest medium fire
# 		else:
# 			medium_fire_dists = list(map(dist, FIRE_LOCATIONS[4:6]))
# 			if( np.abs(medium_fire_dists[0] - medium_fire_dists[1]) < 0.1  ):
# 				closest_medium_fire = 4 if random.random() < 0.5 else 5
# 			else:
# 				closest_medium_fire = np.argmin( medium_fire_dists ) + 4
# 			next_closest_medium_fire = 5 if (closest_medium_fire == 4) else 4
# 			if(fire_statuses[closest_medium_fire] > 0.5):
# 				if(not within_epsilon(my_loc, FIRE_LOCATIONS[closest_medium_fire])):
# 					return closest_medium_fire
# 				else:
# 					return NUM_FIRES
# 			else:
# 				if(not within_epsilon(my_loc, FIRE_LOCATIONS[next_closest_medium_fire])):
# 					return next_closest_medium_fire
# 				else:
# 					return NUM_FIRES

# 	def get_actions(self, olist):
# 		return [ self.get_action(o) for o in olist], {}

# ---- Many Agents Many Fires

# 50F30U
# Num Samps: 3000
# AvgDiscounted: 231.776675905 StdDiscounted: 88.0538105052

# 10U20F
# Num Samps: 500
# 68.3250010153 6.04530597015

class test_policy():

	def reset(self, dones = None):
		return

	@property
	def recurrent(self):
		return False

	def get_action(self, obs):

		my_loc = obs[0:2]		
		fire_statuses = obs[2*NUM_AGENTS : 2*NUM_AGENTS + NUM_FIRES]

		dist = lambda x: np.linalg.norm( np.array(x) - np.array(my_loc))

		# Find largest live fires

		live_fire_rewards = [ f if fire_statuses[i] > 0.5 else 0. for i, f in enumerate(FIRE_REWARDS)]
		strongest_live_fire = max(live_fire_rewards)
		indicies_slf = np.where(np.array(live_fire_rewards) == strongest_live_fire)[0].tolist()

		# find closest fire loc

		slf_locs = [ FIRE_LOCATIONS[i] for i in indicies_slf ]

		# if we're at any of these fires, stay
		at_slf_locs = [within_epsilon(i, my_loc) for i in slf_locs]
		if(any(at_slf_locs)):
			# stay
			return NUM_FIRES

		# with probability eps pick a random one
		eps = 0.3 # 30% chance we go to a random stongest live fire
		if(random.random() < eps):
			return indicies_slf[random.randint(0,len(indicies_slf) - 1)]

		# otherwise pick the closest one
		else:
			distances = [distance(i, my_loc) for i in slf_locs]
			min_index, min_value = min(enumerate(distances), key=operator.itemgetter(1))
			return indicies_slf[min_index]

	def get_actions(self, olist):
		return [ self.get_action(o) for o in olist], {}

