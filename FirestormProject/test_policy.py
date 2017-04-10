import numpy as np
import random

class test_policy():

	@property
	def recurrent(self):
		return False

	def reset(self, dones = None):
		return

	def get_actions(self, olist):
		actions = [self.get_action(o) for o in olist]
		return actions, dict(probs = [0]*len(olist))

	def __init__(self):
		return

	def get_action(self, obs):
		if obs == [None]:
			return None
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




from eventdriven.EDhelpers import variable_discount_cumsum, ed_dec_rollout, ed_simpy_dec_rollout
from numpy import std, mean
from sandbox.rocky.tf.envs.base import TfEnv
import time

def path_discounted_returns(env, gamma, num_traj, policy = test_policy(), simpy = False):
	print('Simulating %d Rollouts...' % (num_traj))
	start_time = time.time()

	env = TfEnv(env)

	paths = []
	rollout_times = []
	for i in range(num_traj):
		start_time_r = time.time()
		if(simpy):
			paths.append(ed_simpy_dec_rollout(env, policy))
		else:
			paths.append(ed_dec_rollout(env, policy))
		elapsed_r = time.time() - start_time_r
		rollout_times.append(elapsed_r)



	paths = [item for sublist in paths for item in sublist]

	adr = []

	for path in paths:
		t_sojourn = path["offset_t_sojourn"]
		discount_gamma = np.exp(-gamma*t_sojourn)
		path_adr = variable_discount_cumsum(path["rewards"], discount_gamma)
		avg_discounted_return = path_adr[0]
		adr.append(avg_discounted_return)

	elapsed = time.time() - start_time
	print('Time Elapsed %.2f, or %.7f +- %.7f per rollout' % (elapsed, mean(rollout_times), std(rollout_times) / np.sqrt(num_traj)))

	return mean(adr), std(adr) / np.sqrt(num_traj)









