import numpy as np
from gym import spaces

from rltools import nn
import rltools.util
from rltools.trajutil import RaggedArray, TrajBatch, Trajectory


class Sampler(object):
    """
    Base Sampler class
    """

    def __init__(self, algo, n_timesteps, max_traj_len, timestep_rate, n_timesteps_min,
                 n_timesteps_max, adaptive, enable_rewnorm):
        self.algo = algo
        self.n_timesteps = n_timesteps
        self.max_traj_len = max_traj_len
        self.adaptive = adaptive
        self.rewnorm = (nn.Standardizer if enable_rewnorm else nn.NoOpStandardizer)(1)
        if self.adaptive:
            self.timestep_rate = timestep_rate
            self.n_timesteps_min = n_timesteps_min
            self.n_timesteps_max = n_timesteps_max
        self.n_episodes = 0

    def start(self):
        """Init sampler"""
        raise NotImplementedError()

    def sample(self, itr):
        """Collect samples"""
        raise NotImplementedError()

    def process(self, sess, itr, trajbatch, discount, gae_lambda, baseline):
        B = len(trajbatch)
        trajlens = [len(traj) for traj in trajbatch]
        maxT = max(trajlens)

        rewards_B_T = self.rewnorm.standardize(
            trajbatch.r.padded(fill=0.), centered=False, sess=sess)
        assert not self.algo.discount is None
        qvals_zfilled_B_T = rltools.util.discount(rewards_B_T, discount)
        assert qvals_zfilled_B_T.shape == (B, maxT)
        q = RaggedArray([qvals_zfilled_B_T[i, :len(traj)] for i, traj in enumerate(trajbatch)])
        q_B_T = q.padded(fill=np.nan)  # q vals padded with nans in the end
        assert q_B_T.shape == (B, maxT)

        # Time-dependent baseline
        simplev_B_T = np.tile(np.nanmean(q_B_T, axis=0, keepdims=True), (B, 1))
        assert simplev_B_T.shape == (B, maxT)
        simplev = RaggedArray([simplev_B_T[i, :len(traj)] for i, traj in enumerate(trajbatch)])

        # State-dependent baseline
        v_stacked = baseline.predict(sess, trajbatch)
        assert v_stacked.ndim == 1
        v = RaggedArray(v_stacked, lengths=trajlens)

        # Compare squared loss of value function to that of time-dependent value function
        # Explained variance
        # *_r2 = 1 - var(y-ypred)/var(y)
        # *_r2 = 0 => Useless
        # *_r2 = 1 => Perfect
        # *_r2 < 0 => Worse than useless
        constfunc_prediction_loss = np.var(q.stacked)
        simplev_prediction_loss = np.var(q.stacked - simplev.stacked)
        simplev_r2 = 1. - simplev_prediction_loss / (constfunc_prediction_loss + 1e-8)
        vfunc_prediction_loss = np.var(q.stacked - v_stacked)
        vfunc_r2 = 1. - vfunc_prediction_loss / (constfunc_prediction_loss + 1e-8)

        # XXX HACK
        if vfunc_r2 < 0:
            v = simplev

        # Compute advantage -- GAE(gamma,lambda) estimator
        v_B_T = v.padded(fill=0.)
        v_B_Tp1 = np.concatenate([v_B_T, np.zeros((B, 1))], axis=1)
        #assert v_B_Tp1.shape == (B, maxT + 1)
        delta_B_T = rewards_B_T + discount * v_B_Tp1[:, 1:] - v_B_Tp1[:, :-1]
        adv_B_T = rltools.util.discount(delta_B_T, discount * gae_lambda)
        #assert adv_B_T.shape == (B, maxT)
        adv = RaggedArray([adv_B_T[i, :l] for i, l in enumerate(trajlens)])
        assert np.allclose(adv.padded(fill=0), adv_B_T)

        # Fit for the next time step
        baseline_info = baseline.fit(sess, trajbatch, q.stacked)

        return dict(advantage=adv, qval=q, v_r=vfunc_r2, tv_r=simplev_r2), baseline_info

    def stop(self):
        raise NotImplementedError()


def centrollout(env, policy, max_traj_len, action_space):
    policy.reset()
    obs, actions, actiondists, rewards = [], [], [], []
    traj_info_list = []
    obs.append(np.c_[env.reset()].ravel()[None, ...].copy())

    for itr in range(max_traj_len):
        a, adist = policy.sample_actions(obs[-1])
        actions.append(a)
        actiondists.append(adist)
        if isinstance(action_space, spaces.Discrete):
            ndim = 1 if not hasattr(action_space, 'ndim') else action_space.ndim
            assert a.ndim == 2 and a.dtype in (np.int32, np.int64)
            if ndim == 1:
                o2, r, done, info = env.step(actions[-1][0, 0])  # XXX
            else:
                o2, r, done, info = env.step(actions[-1][0, :ndim])  # XXX
        else:
            o2, r, done, info = env.step(actions[-1][0])

        if isinstance(r, list) or isinstance(r, np.ndarray):
            assert (r == r[0]).all()
            rewards.append(r[0])
        else:
            rewards.append(r)

        if info:
            traj_info_list.append(info)

        if done:
            break
        if itr != max_traj_len - 1:
            obs.append(np.c_[o2].ravel()[None, ...])

    traj_info = rltools.util.stack_dict_list(traj_info_list)
    obs_T_Do = np.concatenate(obs)
    assert obs_T_Do.shape[0] == len(obs), '{} != {}'.format(obs_T_Do.shape, len(obs))
    adist_T_Pa = np.concatenate(actiondists)
    assert adist_T_Pa.ndim == 2 and adist_T_Pa.shape[0] == len(obs)
    a_T_Da = np.concatenate(actions)
    assert a_T_Da.shape[0] == len(obs)
    r_T = np.asarray(rewards)
    assert r_T.shape == (len(obs),)

    return Trajectory(obs_T_Do, adist_T_Pa, a_T_Da, r_T, traj_info)


def get_lists(nl, na):
    l = []
    for i in range(nl):
        l.append([[] for j in range(na)])
    return l


def decrollout(env, policy, max_traj_len, action_space):
    assert not isinstance(policy, list)
    policy.reset(dones=[True] * len(env.agents))
    trajs = []
    old_obs = env.reset()
    obs, actions, actiondists, rewards = get_lists(4, len(env.agents))
    traj_info_list = []

    for itr in range(max_traj_len):
        agent_actions, adist_list = policy.sample_actions(np.asarray(old_obs))
        comp_actions = np.array(agent_actions)
        for i, agent_obs in enumerate(old_obs):
            obs[i].append(np.expand_dims(agent_obs, 0))
            actions[i].append(agent_actions[i])
            actiondists[i].append(adist_list[i])

        if isinstance(action_space, spaces.Discrete):
            new_obs, r, done, info = env.step(comp_actions[:, 0])
        else:
            new_obs, r, done, info = env.step(comp_actions)

        if info:
            traj_info_list.append(info)

        for i, o in enumerate(old_obs):
            if o is None:
                continue
            rewards[i].append(r[i])

        old_obs = new_obs

        if done:
            break

    traj_info = rltools.util.stack_dict_list(traj_info_list)
    for agnt in range(len(env.agents)):
        obs_T_Do = np.concatenate(obs[agnt])
        adist_T_Pa = np.concatenate(np.expand_dims(np.asarray(actiondists[agnt]), 0))
        a_T_Da = np.concatenate(np.expand_dims(np.asarray(actions[agnt]), 0))
        r_T = np.asarray(rewards[agnt])
        trajs.append(Trajectory(obs_T_Do, adist_T_Pa, a_T_Da, r_T, traj_info))

    return trajs


def concrollout(env, policies, max_traj_len, action_space):
    assert len(policies) == len(env.agents)

    for policy in policies:
        policy.reset()
    trajs = []
    old_obs = env.reset()
    obs, actions, actiondists, rewards = get_lists(4, len(env.agents))
    traj_info_list = []
    for itr in range(max_traj_len):
        agent_actions, adist_list = [], []
        for i, agent_obs in enumerate(old_obs):
            act, adist = policies[i].sample_actions(np.expand_dims(agent_obs, 0))
            agent_actions.append(act)
            adist_list.append(adist)

        comp_actions = np.array(agent_actions)
        for i, agent_obs in enumerate(old_obs):
            obs[i].append(np.expand_dims(agent_obs, 0))
            actions[i].append(agent_actions[i])
            actiondists[i].append(adist_list[i])

        if isinstance(action_space, spaces.Discrete):
            new_obs, r, done, info = env.step(comp_actions[:, 0, 0])
        else:
            new_obs, r, done, info = env.step(comp_actions)

        if info:  # XXX what if some infos are none in a traj?
            traj_info_list.append(info)

        for i, o in enumerate(old_obs):
            if o is None:
                continue
            rewards[i].append(r[i])

        old_obs = new_obs

        if done:
            break

    traj_info = rltools.util.stack_dict_list(traj_info_list)
    for agnt in range(len(env.agents)):
        a_T_Da = np.concatenate(np.asarray(actions[agnt]))
        r_T = np.asarray(rewards[agnt])
        trajs.append(Trajectory(obs_T_Do, adist_T_Pa, a_T_Da, r_T, traj_info))

    return trajs
