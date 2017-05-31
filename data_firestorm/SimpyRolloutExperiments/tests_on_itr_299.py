# Tests conducted on simpy_rollout_fire_smdp with fixed steps. 
# Tests conducted only on itr_299, so not very representitive.

# with tf.Session() as sess:
	# 	obj = joblib.load('./data/experiment_2017_04_10_11_21_38_simpy_rollout/itr_149.pkl')
	# 	policy = obj['policy']
	# 	print('ED Learned Policy')
	# 	print(path_discounted_returns(env = env, num_traj = num_trajs_sim, gamma = GAMMA, policy = policy, simpy = True))

	# tf.reset_default_graph()
	# with tf.Session() as sess:
	# 	from FirestormProject.fixedstep_fire_smdp import FixedStepFireExtinguishingEnv
	# 	obj = joblib.load('./data/n_parallel1_simpymayhavebugs/experiment_2017_04_07_23_27_23_fixed_10e-1/itr_149.pkl')
	# 	policy = obj['policy']
	# 	print('Fixed-Step 0.1 Learned Policy')
	# 	print(path_discounted_returns(env = env, num_traj = num_trajs_sim, gamma = GAMMA, policy = policy, simpy = True))

	# tf.reset_default_graph()
	# with tf.Session() as sess:
	# 	from FirestormProject.fixedstep_fire_smdp import FixedStepFireExtinguishingEnv
	# 	obj = joblib.load('./data/n_parallel1_simpymayhavebugs/experiment_2017_04_07_22_13_22_fixed_10e-0p5/itr_149.pkl')
	# 	policy = obj['policy']
	# 	print('Fixed-Step 0.32 Learned Policy')
	# 	print(path_discounted_returns(env = env, num_traj = num_trajs_sim, gamma = GAMMA, policy = policy, simpy = True))

	# tf.reset_default_graph()
	# with tf.Session() as sess:
	# 	from FirestormProject.fixedstep_fire_smdp import FixedStepFireExtinguishingEnv
	# 	obj = joblib.load('./data/n_parallel1_simpymayhavebugs/experiment_2017_04_07_21_44_55_fixed_10e0/itr_149.pkl')
	# 	policy = obj['policy']
	# 	print('Fixed-Step 1.0 Learned Policy')
	# 	print(path_discounted_returns(env = env, num_traj = num_trajs_sim, gamma = GAMMA, policy = policy, simpy = True))

	# tf.reset_default_graph()
	# with tf.Session() as sess:
	# 	from FirestormProject.fixedstep_fire_smdp import FixedStepFireExtinguishingEnv
	# 	obj = joblib.load('./data/n_parallel1_simpymayhavebugs/experiment_2017_04_07_20_42_19_fixed_10e0p5/itr_149.pkl')
	# 	policy = obj['policy']
	# 	print('Fixed-Step 3.2 Learned Policy')
	# 	print(path_discounted_returns(env = env, num_traj = num_trajs_sim, gamma = GAMMA, policy = policy, simpy = True))

	# tf.reset_default_graph()
	# with tf.Session() as sess:
	# 	from FirestormProject.fixedstep_fire_smdp import FixedStepFireExtinguishingEnv
	# 	obj = joblib.load('./data/n_parallel1_simpymayhavebugs/experiment_2017_04_07_18_44_35_fixed_10e1/itr_149.pkl')
	# 	policy = obj['policy']
	# 	print('Fixed-Step 10.0 Learned Policy')
	# 	print(path_discounted_returns(env = env, num_traj = num_trajs_sim, gamma = GAMMA, policy = policy, simpy = True))

	# tf.reset_default_graph()
	# with tf.Session() as sess:
	# 	obj = joblib.load('./data/experiment_2017_04_10_11_21_38_simpy_rollout/itr_299.pkl')
	# 	policy = obj['policy']
	# 	print('ED Learned Policy')
	# 	print(path_discounted_returns(env = env, num_traj = num_trajs_sim, gamma = GAMMA, policy = policy, simpy = True))
	# 	# (3.5252302193830318, 0.043178665568979431)

	# tf.reset_default_graph()
	# with tf.Session() as sess:
	# 	obj = joblib.load('./data/experiment_2017_04_12_18_35_01_simpy_rollout_dt10e1/itr_299.pkl')
	# 	policy = obj['policy']
	# 	print('Fixed-Step ED 10.0 Learned Policy')
	# 	print(path_discounted_returns(env = env, num_traj = num_trajs_sim, gamma = GAMMA, policy = policy, simpy = True))
	# 	# (1.5824974091311366, 0.040834568322337998)

	# tf.reset_default_graph()
	# with tf.Session() as sess:
	# 	obj = joblib.load('./data/experiment_2017_04_12_20_06_16_simpy_rollout_dt10e0.5/itr_299.pkl')
	# 	policy = obj['policy']
	# 	print('Fixed-Step ED 3.2 Learned Policy')
	# 	print(path_discounted_returns(env = env, num_traj = num_trajs_sim, gamma = GAMMA, policy = policy, simpy = True))
	# 	# (3.3845543345031381, 0.041592195794508034)


	# tf.reset_default_graph()
	# with tf.Session() as sess:
	# 	obj = joblib.load('./data/experiment_2017_04_12_20_58_13_simpy_rollout_dt10e0/itr_299.pkl')
	# 	policy = obj['policy']
	# 	print('Fixed-Step ED 1.0 Learned Policy')
	# 	print(path_discounted_returns(env = env, num_traj = num_trajs_sim, gamma = GAMMA, policy = policy, simpy = True))

	# 	# (3.4167419631484468, 0.042274181774593585)

	# tf.reset_default_graph()
	# with tf.Session() as sess:
	# 	obj = joblib.load('./data/experiment_2017_04_12_22_09_47_simpy_rollout_dt10e-0.5/itr_299.pkl')
	# 	policy = obj['policy']
	# 	print('Fixed-Step ED 0.32 Learned Policy')
	# 	print(path_discounted_returns(env = env, num_traj = num_trajs_sim, gamma = GAMMA, policy = policy, simpy = True))
	# 	# (3.4381591675362917, 0.043531627634683587)

	tf.reset_default_graph()
	with tf.Session() as sess:
		obj = joblib.load('./data/experiment_2017_04_12_23_20_03_simpy_rollout_dt10e-1/itr_299.pkl')
		policy = obj['policy']
		print('Fixed-Step ED 0.1 Learned Policy')
		print(path_discounted_returns(env = env, num_traj = num_trajs_sim, gamma = GAMMA, policy = policy, simpy = True))
		# (3.400896427877143, 0.022225470246890311)






# ED Learned Policy
# Env is of type  <class '__main__.FireExtinguishingEnv'>
# Policy is of type  <class 'sandbox.rocky.tf.policies.categorical_mlp_policy.CategoricalMLPPolicy'>
# Simulating 2000 Rollouts...
# 100% (2000 of 2000) |#####################################################################################################################| Elapsed Time: 0:02:07 Time: 0:02:07
# Time Elapsed 127.24, or 0.0634584 +- 0.0010995 per rollout
# (3.489098718370653, 0.02283972092461033)
# Fixed-Step ED 3.2 Learned Policy
# Env is of type  <class '__main__.FireExtinguishingEnv'>
# Policy is of type  <class 'sandbox.rocky.tf.policies.categorical_mlp_policy.CategoricalMLPPolicy'>
# Simulating 2000 Rollouts...
# 100% (2000 of 2000) |#####################################################################################################################| Elapsed Time: 0:02:27 Time: 0:02:27
# Time Elapsed 148.02, or 0.0738183 +- 0.0005699 per rollout
# (3.4277401915106687, 0.021237115667375413)
# Fixed-Step ED 1.0 Learned Policy
# Env is of type  <class '__main__.FireExtinguishingEnv'>
# Policy is of type  <class 'sandbox.rocky.tf.policies.categorical_mlp_policy.CategoricalMLPPolicy'>
# Simulating 2000 Rollouts...
# 100% (2000 of 2000) |#####################################################################################################################| Elapsed Time: 0:02:26 Time: 0:02:26
# Time Elapsed 146.22, or 0.0728839 +- 0.0008591 per rollout
# (3.4558629997592609, 0.021581331009463852)
# Fixed-Step ED 0.32 Learned Policy
# Env is of type  <class '__main__.FireExtinguishingEnv'>
# Policy is of type  <class 'sandbox.rocky.tf.policies.categorical_mlp_policy.CategoricalMLPPolicy'>
# Simulating 2000 Rollouts...
# 100% (2000 of 2000) |#####################################################################################################################| Elapsed Time: 0:02:26 Time: 0:02:26
# Time Elapsed 146.16, or 0.0728930 +- 0.0014716 per rollout
# (3.4864932188823077, 0.022113559841211779)
