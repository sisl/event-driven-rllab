from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from EDFirestorm.EDfirestorm_singleagent_env import EDFirestorm_SingleAgent_Env
from EDFirestorm.EDfirestorm_singleagent_env import NonEDFirestorm_SingleAgent_Env
from rllab.envs.normalized_env import normalize
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy

# DDPG
from rllab.algos.cem import CEM
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction
from rllab.exploration_strategies.ou_strategy import OUStrategy

import rllab.misc.logger as logger


nonED = True

if(nonED):
	# logger.add_tabular_output('./NonED.log')
	env = NonEDFirestorm_SingleAgent_Env()
	discount = env.discount
	env = normalize(env)
	policy = CategoricalMLPPolicy(
	    env_spec=env.spec,
	)

	baseline = LinearFeatureBaseline(env_spec=env.spec)
	algo = TRPO(
	    env=env,
	    policy=policy,
	    baseline=baseline,
	    discount=discount,
	    n_itr=75
	)
else:
	logger.add_tabular_output('./ED.log')
	env = normalize(EDFirestorm_SingleAgent_Env())
	policy = CategoricalMLPPolicy(
	    env_spec=env.spec,
	)

	baseline = LinearFeatureBaseline(env_spec=env.spec)
	algo = TRPO(
	    env=env,
	    policy=policy,
	    baseline=baseline,
	    discount=0.999,
	    n_itr=75
	)

algo.train()