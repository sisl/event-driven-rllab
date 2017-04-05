import numpy as np
from rllab.envs.base import Env
from rllab.spaces import Discrete
from rllab.envs.base import Step
from rllab.core.serializable import Serializable



class GridWorldEnv(Env, Serializable):

    def __init__(self, desc='4x4'):
        Serializable.quick_init(self, locals())


    def reset(self):
        self.state = 1
        return self.state


    def step(self, action):
        if(action == 0):
            next_state = self.state - 1
        elif(action == 1):
            next_state = self.state + 1
        else:
            raise NotImplementedError

        rewards = [1, 0, -1, 3]
        reward = rewards[next_state]

        dones = [True, False, False, True]
        done = dones[next_state]

        self.state = next_state

        return Step(observation=self.state, reward=reward, done=done)



    @property
    def action_space(self):
        return Discrete(2)

    @property
    def observation_space(self):
        return Discrete(4)


if __name__ == "__main__":

    env = GridWorldEnv()


    from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
    from sandbox.rocky.tf.core.network import MLP
    from sandbox.rocky.tf.envs.base import TfEnv
    from sandbox.rocky.tf.algos.trpo import TRPO
    import tensorflow as tf
    env = TfEnv(env)


    from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
    
    policy = CategoricalMLPPolicy(env_spec=env.spec, name = "policy")

    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        whole_paths=True,
        max_path_length=50,
        n_itr=40,
        discount=0.40
    )
    algo.train()

