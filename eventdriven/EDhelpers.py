## Helper file containing all functions required for 
#  using Event-Drive (multi/single-agent) environments
#  with rllab

# Definitions:
#   ED: Event-Driven
#   GSMDP: Generalized Semi-Markov Decision Process

# General imports
import numpy as np
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.algos import util
import rllab.misc.logger as logger

import itertools

import pdb


## ed_dec_rollout
# Adapted from etotheipi/MADRL dec_rollout
# Performs a decentralized rollout handling the MAED protocol
# See MAED (Multi-Agent Event-Driven) protocol in the README.txt

# Dependencies: Requires agents to be a sandbox.rocky.tf.policies policy

import time

def ed_dec_rollout(env, agents, max_path_length=np.inf, animated=False, speedup=1):
	if(agents.recurrent):
		assert isinstance(agents, GSMDPRecurrentPolicy), 'Recurrent policy is not a GSMDP class'

	"""Decentralized rollout"""
	n_agents = len(env.agents)
	observations = [[] for _ in range(n_agents)]
	actions = [[] for _ in range(n_agents)]
	rewards = [[] for _ in range(n_agents)]
	agent_infos = [[] for _ in range(n_agents)]
	env_infos = [[] for _ in range(n_agents)]
	offset_t_sojourn = [[] for _ in range(n_agents)]
	olist = env.reset()
	assert len(olist) == n_agents, "{} != {}".format(len(olist), n_agents)


	

	agents.reset(dones=[True for _ in range(n_agents)])
	path_length = 0
	if animated:
		env.render()
	while path_length < max_path_length:
		agents_to_act = [i for i,j in enumerate(olist) if j != [None]*len(j)] 
		if(not agents.recurrent):
			alist, agent_info_list = agents.get_actions([olist[i] for i in agents_to_act])
			agent_info_list = tensor_utils.split_tensor_dict_list(agent_info_list)
		else:
			alist, agent_info_list = agents.get_actions(olist)
			alist = [a for a in alist if a != None]
			agent_info_list = tensor_utils.split_tensor_dict_list(agent_info_list)
			agent_info_list = [ainfo for i, ainfo in enumerate(agent_info_list) if i in agents_to_act]



		next_actions = [None]*n_agents # will fill in in the loop


		# For each agent
		for ind, o in enumerate([olist[j] for j in agents_to_act]):
			# ind refers to non-None indicies
			# i refers to indices with Nones
			i = agents_to_act[ind]
			observations[i].append(env.observation_space.flatten(o))
			# observations[i].append(o) # REMOVE THIS AND UNCOMMENT THE ABOVE LINE
			actions[i].append(env.action_space.flatten(alist[ind]))
			next_actions[i] = alist[ind]
			if agent_info_list is None:
				agent_infos[i].append({})
			else:
				agent_infos[i].append(agent_info_list[ind])

		# take next actions
		next_olist, rlist, d, env_info = env.step(np.asarray(next_actions))

		# update sojourn time (we should associate ts from next_olist to r, not current)
		
		for i, r in enumerate(rlist):
			if r is None: continue
			# skip reward if agent has not acted yet
			if( len(observations[i]) > 0 ):
				rewards[i].append(r)
				offset_t_sojourn[i].append(env.observation_space.flatten(next_olist[i])[-1])
				env_infos[i].append(env_info)
		path_length = max( [len(o) for o in observations] ) 
		if d:
			break
		olist = next_olist
		if animated:
			env.render()
			timestep = 0.05
			time.sleep(timestep / speedup)

	if(path_length == max_path_length):
		# probably have some paths that aren't the right length
		for ind, o in enumerate(observations):
			r = rewards[ind]
			if(len(o) > len(r)):
				assert(len(o) <= (len(r) + 1)), \
					'len(o) %d, len(r) %d' % (len(o), len(r))
				# delete last elem of obs, actions, agent infos
				del observations[ind][-1]
				del actions[ind][-1]
				del agent_infos[ind][-1]


	if animated:
		env.render()

	# remove empty agent trajectories
	observations = [ o for o in observations if len(o) > 0]
	actions = [ a for a in actions if len(a) > 0]
	rewards = [ r for r in rewards if len(r) > 0]
	agent_infos = [i for i in agent_infos if len(i) > 0]
	env_infos = [e for e in env_infos if len(e) > 0]


	return [
		dict(
			observations=tensor_utils.stack_tensor_list(observations[i]),
			actions=tensor_utils.stack_tensor_list(actions[i]),
			rewards=tensor_utils.stack_tensor_list(rewards[i]),
			agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos[i]),
			env_infos=tensor_utils.stack_tensor_dict_list(env_infos[i]),
			offset_t_sojourn=tensor_utils.stack_tensor_list(offset_t_sojourn[i]),) for i in range(n_agents)
	]

## Parallel Sampler functions
# _worker_collect_path_one_env calls ed_dec_rollout
# sample_paths calls _worker_collect_path_one_env

from rllab.sampler import parallel_sampler
from rllab.sampler.stateful_pool import singleton_pool, SharedGlobal
from rllab.misc import ext
from rllab.misc import logger

def _worker_collect_path_one_env(G, max_path_length, ma_mode, scope=None):
	G = parallel_sampler._get_scoped_G(G, scope)
	paths = ed_dec_rollout(G.env, G.policy, max_path_length)
	lengths = [len(path['rewards']) for path in paths]
	return paths, sum(lengths)

def sample_paths(
		policy_params,
		max_samples,
		max_path_length=np.inf,
		env_params=None,
		scope=None):
	"""
	:param policy_params: parameters for the policy. This will be updated on each worker process
	:param max_samples: desired maximum number of samples to be collected. The actual number of collected samples
	might be greater since all trajectories will be rolled out either until termination or until max_path_length is
	reached
	:param max_path_length: horizon / maximum length of a single trajectory
	:return: a list of collected paths
	"""
	singleton_pool.run_each(
		parallel_sampler._worker_set_policy_params,
		[(policy_params, scope)] * singleton_pool.n_parallel
	)
	if env_params is not None:
		singleton_pool.run_each(
			parallel_sampler._worker_set_env_params,
			[(env_params, scope)] * singleton_pool.n_parallel
		)
	return singleton_pool.run_collect(
		#_worker_collect_one_path,
		_worker_collect_path_one_env,
		threshold=max_samples,
		args=(max_path_length, scope),
		show_prog_bar=True
	)


## GSMDPSampler
#  Subclass of sampler, superclass of GSMDPBatchSampler

def variable_discount_cumsum(x,discount):
    # Same as discounted cumsum but uses different discount rate for each step
    # y[t] = x[t] + discount[t] * y[y+1], y[N] = x[N]
    x = x[::-1]
    discount = discount[::-1]
    y = np.zeros(x.shape)
    for i in range(x.shape[0]):
        if(i==0):
            y[i] = x[i]
        else:
            y[i] = x[i] + discount[i]*y[i-1]
    return y[::-1]

from rllab.sampler.base import Sampler

class GSMDPSampler(Sampler):
	def __init__(self, algo):
		"""
		:type algo: BatchPolopt
		"""
		self.algo = algo

	def process_samples(self, itr, paths):
		# IMPORTANT:
		# Rewards accrued from a_t to a_t+1 are expected to be discounted by 
		# the environment to values at time t

		#paths = list(itertools.chain.from_iterable(paths))

		baselines = []
		returns = []

		if hasattr(self.algo.baseline, "predict_n"):
			all_path_baselines = self.algo.baseline.predict_n(paths)
		else:
			all_path_baselines = [self.algo.baseline.predict(path) for path in paths]

		for idx, path in enumerate(paths):
			t_sojourn = path["offset_t_sojourn"]
			gamma = self.algo.discount
			lamda = self.algo.gae_lambda
			discount_gamma = np.exp(-gamma*t_sojourn)
			discount_gamma_lambda = np.exp(-gamma*lamda*t_sojourn)
			path_baselines = np.append(all_path_baselines[idx], 0)
			if(len(path["rewards"]) != len(t_sojourn)):
				# TODO HANDLE INFINITE HORIZON GAMES
				pdb.set_trace()
			deltas = path["rewards"] + \
					 discount_gamma * path_baselines[1:] - \
					 path_baselines[:-1]
			path["advantages"] = variable_discount_cumsum(
				deltas, discount_gamma_lambda)
			path["returns"] = variable_discount_cumsum(path["rewards"], discount_gamma)
			baselines.append(path_baselines[:-1])
			returns.append(path["returns"])

		ev = special.explained_variance_1d(
			np.concatenate(baselines),
			np.concatenate(returns)
		)

		if not self.algo.policy.recurrent:
			observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
			actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
			rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
			returns = tensor_utils.concat_tensor_list([path["returns"] for path in paths])
			advantages = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])
			env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
			agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

			if self.algo.center_adv:
				advantages = util.center_advantages(advantages)

			if self.algo.positive_adv:
				advantages = util.shift_advantages_to_positive(advantages)

			average_discounted_return = \
				np.mean([path["returns"][0] for path in paths])

			undiscounted_returns = [sum(path["rewards"]) for path in paths]

			ent = np.mean(self.algo.policy.distribution.entropy(agent_infos))

			samples_data = dict(
				observations=observations,
				actions=actions,
				rewards=rewards,
				returns=returns,
				advantages=advantages,
				env_infos=env_infos,
				agent_infos=agent_infos,
				paths=paths,
			)

		else:
			max_path_length = max([len(path["advantages"]) for path in paths])

			# make all paths the same length (pad extra advantages with 0)
			obs = [path["observations"] for path in paths]
			obs = tensor_utils.pad_tensor_n(obs, max_path_length)

			if self.algo.center_adv:
				raw_adv = np.concatenate([path["advantages"] for path in paths])
				adv_mean = np.mean(raw_adv)
				adv_std = np.std(raw_adv) + 1e-8
				adv = [(path["advantages"] - adv_mean) / adv_std for path in paths]
			else:
				adv = [path["advantages"] for path in paths]

			adv = np.asarray([tensor_utils.pad_tensor(a, max_path_length) for a in adv])

			actions = [path["actions"] for path in paths]
			actions = tensor_utils.pad_tensor_n(actions, max_path_length)

			rewards = [path["rewards"] for path in paths]
			rewards = tensor_utils.pad_tensor_n(rewards, max_path_length)

			returns = [path["returns"] for path in paths]
			returns = tensor_utils.pad_tensor_n(returns, max_path_length)

			agent_infos = [path["agent_infos"] for path in paths]
			agent_infos = tensor_utils.stack_tensor_dict_list(
				[tensor_utils.pad_tensor_dict(p, max_path_length) for p in agent_infos]
			)

			env_infos = [path["env_infos"] for path in paths]
			env_infos = tensor_utils.stack_tensor_dict_list(
				[tensor_utils.pad_tensor_dict(p, max_path_length) for p in env_infos]
			)

			valids = [np.ones_like(path["returns"]) for path in paths]
			valids = tensor_utils.pad_tensor_n(valids, max_path_length)

			average_discounted_return = \
				np.mean([path["returns"][0] for path in paths])

			undiscounted_returns = [sum(path["rewards"]) for path in paths]

			ent = np.sum(self.algo.policy.distribution.entropy(agent_infos) * valids) / np.sum(valids)

			samples_data = dict(
				observations=obs,
				actions=actions,
				advantages=adv,
				rewards=rewards,
				returns=returns,
				valids=valids,
				agent_infos=agent_infos,
				env_infos=env_infos,
				paths=paths,
			)

		logger.log("fitting baseline...")
		if hasattr(self.algo.baseline, 'fit_with_samples'):
			self.algo.baseline.fit_with_samples(paths, samples_data)
		else:
			self.algo.baseline.fit(paths)
		logger.log("fitted")

		logger.record_tabular('Iteration', itr)
		logger.record_tabular('AverageDiscountedReturn',
							  average_discounted_return)
		logger.record_tabular('AverageReturn', np.mean(undiscounted_returns))
		logger.record_tabular('ExplainedVariance', ev)
		logger.record_tabular('NumTrajs', len(paths))
		logger.record_tabular('Entropy', ent)
		logger.record_tabular('Perplexity', np.exp(ent))
		logger.record_tabular('StdReturn', np.std(undiscounted_returns))
		logger.record_tabular('MaxReturn', np.max(undiscounted_returns))
		logger.record_tabular('MinReturn', np.min(undiscounted_returns))

		return samples_data

## GSMDPBatchSampler
# Subclass of GSMDPBatchSampler, passed to rllab algo
class GSMDPBatchSampler(GSMDPSampler):
	def __init__(self, algo):
		"""
		:type algo: BatchPolopt
		"""
		self.algo = algo

	def start_worker(self):
		parallel_sampler.populate_task(self.algo.env, self.algo.policy, scope=self.algo.scope)

	def shutdown_worker(self):
		parallel_sampler.terminate_task(scope=self.algo.scope)

	def obtain_samples(self, itr):
		cur_params = self.algo.policy.get_param_values()
		paths = sample_paths(
			policy_params=cur_params,
			max_samples=self.algo.batch_size,
			max_path_length=self.algo.max_path_length,
			scope=self.algo.scope,
		)
		paths = list(itertools.chain.from_iterable(paths))
		if self.algo.whole_paths:
			return paths
		else:
			paths_truncated = parallel_sampler.truncate_paths(paths, self.algo.batch_size)
			return paths_truncated


## GRU Policies

class GSMDPRecurrentPolicy():
	pass

## CategoricalGRUPolicy
from sandbox.rocky.tf.policies.categorical_gru_policy import CategoricalGRUPolicy
from rllab.misc.overrides import overrides

class GSMDPCategoricalGRUPolicy(CategoricalGRUPolicy, GSMDPRecurrentPolicy):

	# Since the sampler will pass in [None] for an observation that
	# does not need a corresponding action, this function will copy
	# a valid observation into None, but simply dont update the internal
	# state and throw away the action returned

	@overrides
	def get_actions(self, observations):
		# Figure out which agents need valid actions
		agents_to_act = [i for i,j in enumerate(observations) if j != [None]*len(j)]
		agents_not_to_act = [ x for x in list(range(len(observations))) 
			if x not in agents_to_act]

		if(len(agents_to_act) == 0):
			# no agents are acting (shouldn't happen)
			return [None] * len(observations)
		else:
			# copy a valid observation into locations that have [None]
			valid_obs = next(obs for obs in observations if obs != [None]*len(obs))
			observations = [obs if obs != [None]*len(obs) else valid_obs 
				for obs in observations ]

		flat_obs = self.observation_space.flatten_n(observations)
		if self.state_include_action:
			assert self.prev_actions is not None
			try:
				all_input = np.concatenate([
					flat_obs,
					self.prev_actions
				], axis=-1)
			except ValueError:
				all_input = np.concatenate([
					flat_obs,
					self.prev_actions.T
				], axis=-1)

		else:
			all_input = flat_obs
		probs, hidden_vec = self.f_step_prob(all_input, self.prev_hiddens)
		actions = special.weighted_sample_n(probs, np.arange(self.action_space.n))

		#  dont update prev_actions, hidden_vec for non-acting agents
		#  replace those actions with None before returning
		prev_actions = self.prev_actions
		prev_actions_flattened = self.action_space.flatten_n(actions)
		actions = actions.tolist()
		for i in agents_not_to_act:
			hidden_vec[i] = self.prev_hiddens[i]
			prev_actions_flattened[i,:] = prev_actions[i,:]
			actions[i] = None

		self.prev_actions = prev_actions_flattened
		self.prev_hiddens = hidden_vec
			
		agent_info = dict(prob=probs)
		if self.state_include_action:
			agent_info["prev_action"] = np.copy(prev_actions)
		return actions, agent_info

## GaussianGRUPolicy
# TODO DEBUG
from sandbox.rocky.tf.policies.gaussian_gru_policy import GaussianGRUPolicy
from rllab.misc.overrides import overrides

class GSMDPGaussianGRUPolicy(GaussianGRUPolicy, GSMDPRecurrentPolicy):

	# Since the sampler will pass in [None] for an observation that
	# does not need a corresponding action, this function will copy
	# a valid observation into None, but simply dont update the internal
	# state and throw away the action returned

	@overrides
	def get_actions(self, observations):
		# Figure out which agents need valid actions
		agents_to_act = [i for i,j in enumerate(observations) if j != [None]*len(j)]
		agents_not_to_act = [ x for x in list(range(len(observations))) 
			if x not in agents_to_act]

		if(len(agents_to_act) == 0):
			# no agents are acting (shouldn't happen)
			return [None] * len(observations)
		else:
			# copy a valid observation into locations that have [None]
			valid_obs = next(obs for obs in observations if obs != [None]*len(obs))
			observations = [obs if obs != [None]*len(obs) else valid_obs 
				for obs in observations ]

		flat_obs = self.observation_space.flatten_n(observations)
		if self.state_include_action:
			assert self.prev_actions is not None
			all_input = np.concatenate([
				flat_obs,
				self.prev_actions
			], axis=-1)
		else:
			all_input = flat_obs
		means, log_stds, hidden_vec = self.f_step_mean_std(all_input, self.prev_hiddens)
		rnd = np.random.normal(size=means.shape)
		actions = rnd * np.exp(log_stds) + means
		#  dont update prev_actions, hidden_vec for non-acting agents
		#  replace those actions with None before returning
		prev_actions = self.prev_actions
		prev_actions_flattened = self.action_space.flatten_n(actions)
		actions = actions.tolist()
		for i in agents_not_to_act:
			hidden_vec[i] = self.prev_hiddens[i]
			prev_actions_flattened[i,:] = prev_actions[i,:]
			actions[i] = None

		self.prev_actions = prev_actions_flattened
		self.prev_hiddens = hidden_vec

		agent_info = dict(mean=means, log_std=log_stds)
		if self.state_include_action:
			agent_info["prev_action"] = np.copy(prev_actions)

		return actions, agent_info





