from eventdriven.EDhelpers import variable_discount_cumsum, ed_dec_rollout, ed_simpy_dec_rollout
from numpy import std, mean
from sandbox.rocky.tf.envs.base import TfEnv
import random
import time
import progressbar
import numpy as np

class test_policy():

	@property
	def recurrent(self):
		return False

	def reset(self, dones = None):
		return

	def get_actions(self, olist):
		actions = [self.get_action(o) for o in olist]
		return actions, dict(probs = [0]*len(olist))

	def __init__(self, mode):
		self.mode = mode
		return

	def get_action(self, obs):
		if self.mode == 'No-H':
			return 0 # random.randint(0,3)
		elif self.mode == 'T-H':
			headway = obs[0]
			if headway < 0.8 * 6 * 60:
				return 1
			else:
				return 0



def path_discounted_returns(env, gamma, num_traj, simpy = False, policy = test_policy(mode='T-H'), printing = False):
	# print('Env is of type ', type(env))
	# print('Policy is of type ', type(policy))
	if printing: print('Simulating %d Rollouts...' % (num_traj))
	start_time = time.time()

	if(not isinstance(env,TfEnv)):
		env = TfEnv(env)

	paths = []
	rollout_times = []

	if printing: 
		bar = progressbar.ProgressBar()
		iterator = bar(range(num_traj))
	else:
		iterator = range(num_traj)
	for i in iterator:
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
	if printing: print('Time Elapsed %.2f, or %.7f +- %.7f per rollout' % (elapsed, mean(rollout_times), std(rollout_times) / np.sqrt(num_traj)))
	if printing: print('MeanADR: %.3f, StdADR: %.3f' % (mean(adr),std(adr) / np.sqrt(num_traj) ))
	return mean(adr), std(adr) / np.sqrt(num_traj), adr



if __name__ == '__main__':
	main()