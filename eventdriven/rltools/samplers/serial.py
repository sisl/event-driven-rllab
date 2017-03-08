import copy
import random

import numpy as np

from rltools.samplers import Sampler, centrollout, decrollout, concrollout
from rltools.trajutil import TrajBatch, Trajectory


class SimpleSampler(Sampler):

    def __init__(self, algo, n_timesteps, max_traj_len, timestep_rate, n_timesteps_min,
                 n_timesteps_max, adaptive=False, enable_rewnorm=True):
        super(SimpleSampler, self).__init__(algo, n_timesteps, max_traj_len, timestep_rate,
                                            n_timesteps_min, n_timesteps_max, adaptive,
                                            enable_rewnorm)

    def sample(self, sess, itr):
        if self.adaptive and itr > 0 and self.n_timesteps < self.n_timesteps_max:
            if itr % self.timestep_rate == 0:
                self.n_timesteps *= 2

        trajs = []
        timesteps_sofar = 0
        while True:

            traj = centrollout(self.algo.env, self.algo.policy, self.max_traj_len,
                               self.algo.policy.action_space)
            trajs.append(traj)
            timesteps_sofar += len(traj)
            if timesteps_sofar >= self.n_timesteps:
                break

        trajbatch = TrajBatch.FromTrajs(trajs)
        self.n_episodes += len(trajbatch)
        return (trajbatch,
                [('ret', trajbatch.r.padded(fill=0.).sum(axis=1).mean(), float
                 ),  # average return for batch of traj
                 ('batch', len(trajbatch), int),  # batch size                 
                 ('n_episodes', self.n_episodes, int),  # total number of episodes
                 ('avglen', int(np.mean([len(traj) for traj in trajbatch])), int
                 ),  # average traj length
                 ('maxlen', int(np.max([len(traj) for traj in trajbatch])), int),  # max traj length
                 ('minlen', int(np.min([len(traj) for traj in trajbatch])), int),  # min traj length
                 ('ravg', trajbatch.r.stacked.mean(), int
                 )  # avg reward encountered per time step (probably not that useful)
                ] + [(info[0], np.mean(info[1]), float) for info in trajbatch.info])


class DecSampler(Sampler):

    def __init__(self, algo, n_timesteps, max_traj_len, timestep_rate, n_timesteps_min,
                 n_timesteps_max, adaptive=False, enable_rewnorm=True):
        super(DecSampler, self).__init__(algo, n_timesteps, max_traj_len, timestep_rate,
                                         n_timesteps_min, n_timesteps_max, adaptive, enable_rewnorm)

    def sample(self, sess, itr):
        if self.adaptive and itr > 0 and self.n_timesteps < self.n_timesteps_max:
            if itr % self.timestep_rate == 0:
                self.n_timesteps *= 2

        env = self.algo.env
        trajs = []
        timesteps_sofar = 0
        while True:

            ag_trajs = decrollout(self.algo.env, self.algo.policy, self.max_traj_len,
                                  self.algo.policy.action_space)
            trajs.extend(ag_trajs)
            timesteps_sofar += np.sum(map(len, ag_trajs))
            if timesteps_sofar >= self.n_timesteps:
                break

        trajbatch = TrajBatch.FromTrajs(trajs)
        return (trajbatch,
                [('ret', trajbatch.r.padded(fill=0.).sum(axis=1).mean(), float
                 ),  # average return for batch of traj
                 ('batch', len(trajbatch), int),  # batch size                 
                 ('avglen', int(np.mean([len(traj) for traj in trajbatch])), int
                 ),  # average traj length
                 ('maxlen', int(np.max([len(traj) for traj in trajbatch])), int),  # max traj length
                 ('minlen', int(np.min([len(traj) for traj in trajbatch])), int),  # min traj length
                 ('ravg', trajbatch.r.stacked.mean(), int
                 )  # avg reward encountered per time step (probably not that useful)
                ] + [(info[0], np.mean(info[1]), float) for info in trajbatch.info])


class ConcSampler(Sampler):

    def __init__(self, algo, n_timesteps, max_traj_len, timestep_rate, n_timesteps_min,
                 n_timesteps_max, adaptive=False, enable_rewnorm=True):
        super(ConcSampler, self).__init__(algo, n_timesteps, max_traj_len, timestep_rate,
                                          n_timesteps_min, n_timesteps_max, adaptive,
                                          enable_rewnorm)

    def sample(self, sess, itr):
        if self.adaptive and itr > 0 and self.n_timesteps < self.n_timesteps_max:
            if itr % self.timestep_rate == 0:
                self.n_timesteps *= 2

        env = self.algo.env
        timesteps_sofar = 0
        trajslist = [[] for _ in range(len(env.agents))]
        while True:
            ag_trajs = concrollout(self.algo.env, self.algo.policies, self.max_traj_len,
                                   self.algo.policies[0].action_space)
            for agid, agtraj in enumerate(ag_trajs):
                trajslist[agid].append(agtraj)

            timesteps_sofar += len(ag_trajs[0])
            if timesteps_sofar >= self.n_timesteps:
                break

        trajbatches = [TrajBatch.FromTrajs(trajs) for trajs in trajslist]
        self.n_episodes += len(trajbatches[0])
        return (
            trajbatches,
            [('ret', np.mean(
                [trajbatch.r.padded(fill=0.).sum(axis=1).mean() for trajbatch in trajbatches]),
              float),
             ('batch', np.mean([len(trajbatch) for trajbatch in trajbatches]), float),
             ('n_episodes', self.n_episodes, int),  # total number of episodes                 
             ('avglen',
              int(np.mean([len(traj) for traj in trajbatch for trajbatch in trajbatches])), int),
             ('maxlen', int(np.max([len(traj) for traj in trajbatch for trajbatch in trajbatches])),
              int),  # max traj length
             ('minlen', int(np.min([len(traj) for traj in trajbatch for trajbatch in trajbatches])),
              int),  # min traj length
             ('ravg', np.mean([trajbatch.r.stacked.mean() for trajbatch in trajbatches]), float)] +
            [(info[0], np.mean(info[1]), float) for info in trajbatches[0].info])


class ImportanceWeightedSampler(SimpleSampler):
    """
    Alternate between sampling iterations using simple sampler and importance sampling iterations

    Does not work with a NN value function baseline

    #FIXME
    """

    def __init__(self, algo, n_timesteps, max_traj_len, timestep_rate, n_timesteps_min,
                 n_timesteps_max, adaptive=False, enable_rewnorm=True, n_backtrack='all',
                 randomize_draw=False, n_pretrain=0, skip_is=False, max_is_ratio=0):
        """
        n_backtrack: number of past policies to update from
        n_pretrain: iteration number until which to only do importance sampling
        skip_is: whether to skip doing alternate importance sampling after pretraining
        max_is_ratio: maximum importance sampling ratio (thresholding)
        """
        self.n_backtrack = n_backtrack
        self.randomize_draw = randomize_draw
        self.n_pretrain = n_pretrain
        self.skip_is = skip_is
        self.max_is_ratio = max_is_ratio
        self._hist = []
        self._is_itr = 0
        super(ImportanceWeightedSampler, self).__init__(algo, n_timesteps, max_traj_len,
                                                        timestep_rate, n_timesteps_min,
                                                        n_timesteps_max, adaptive, enable_rewnorm)
        assert not self.adaptive, "Can't use adaptive sampling with importance weighted for now"  # TODO needed?

    @property
    def history(self):
        return self._hist

    def add_history(self, trajbatch):
        self.history.append(trajbatch)

    def get_history(self, n_past='all'):
        if n_past == 'all':
            return self.history
        assert isinstance(n_past, int)
        return self.history[-min(n_past, len(self.history)):]

    def sample(self, sess, itr):
        # Importance sampling for first few iterations
        if itr < self.n_pretrain:
            trajbatch = self.is_sample(sess, itr)
            return trajbatch

        # Alternate between importance sampling and actual sampling
        # Data logs will be messy TODO
        if self._is_itr and not self.skip_is:
            trajbatch, batch_info = self.is_sample(sess, itr)
        else:
            trajbatch, batch_info = super(ImportanceWeightedSampler, self).sample(sess, itr)
            if not self.skip_is:
                self.add_history(trajbatch)

        self._is_itr = (self._is_itr + 1) % 2

        return trajbatch, batch_info

    def is_sample(self, sess, itr):
        rettrajs = []
        for hist_trajbatch in self.get_history(self.n_backtrack):
            n_trajs = len(hist_trajbatch)
            n_samples = min(n_trajs, self.batch_size)

            if self.randomize_draw:
                samples = random.sample(hist_trajbatch, n_samples)
            elif hist_trajbatch:
                # Random start
                start = random.randint(0, n_trajs - n_samples)
                samples = hist_trajbatch[start:start + n_samples]

            samples = copy.deepcopy(samples)  # Avoid overwriting

            for traj in samples:
                # What the current policy would have done
                _, adist_T_Pa = self.algo.policy.sample_actions(traj.obs_T_Do)
                # What the older policy did
                hist_adist_T_Pa = traj.adist_T_Pa

                assert traj.adist_T_Pa.shape == adist_T_Pa.shape
                # Use newer policy distribution
                traj.adist_T_Pa = adist_T_Pa

                # Log probabilities of actions using previous and current
                logprob_curr = self.algo.policy.distribution.log_density(adist_T_Pa, traj.a_T_Da)
                logprob_hist = self.algo.policy.distribution.log_density(hist_adist_T_Pa,
                                                                         traj.a_T_Da)
                # Importance sampling ratio
                is_ratio = np.exp(logprob_curr.sum() - logprob_hist.sum())

                # Thresholding
                if self.max_is_ratio > 0:
                    is_ratio = min(is_ratio, self.max_is_ratio)

                # Weight the rewards accordingly
                traj.r_T *= is_ratio

            rettrajs.extend(samples)
            # Pack them back
        if len(rettrajs) > self.batch_size:
            rettrajs = random.sample(rettrajs, self.batch_size)
            rettrajbatch = TrajBatch.FromTrajs(rettrajs)

        batch_info = [('ret', rettrajbatch.r.padded(fill=0.).sum(axis=1).mean(),
                       float),  # average return for batch of traj
                      ('avglen', int(np.mean([len(traj) for traj in rettrajbatch])),
                       int),  # average traj length
                      ('ravg', rettrajbatch.r.stacked.mean(),
                       int)  # avg reward encountered per time step (probably not that useful)
                     ]

        return rettrajbatch, batch_info


class ExperienceReplay(Sampler):

    def __init__(self, algo, max_traj_len, batch_size, min_batch_size, max_batch_size, batch_rate,
                 adaptive=False, initial_exploration=5000, max_experience=10000):
        super(ExperienceReplay, self).__init__(algo, max_traj_len, batch_size, min_batch_size,
                                               max_batch_size, batch_rate, adaptive)
        self._observations = np.zeros((max_experience,) + self.algo.env.observation_space.shape)
        self._actions = np.zeros(max_experience, dtype=np.int32)
        self._rewards = np.zeros(max_experience)
        self._next_observations = np.zeros((max_experience,) +
                                           self.algo.env.observation_space.shape)
        self._terminals = np.zeros(max_experience, dtype=np.bool)

        self.head = 0
        self.replay_size = 0
        self.initial_exploration = initial_exploration
        self.max_experience = max_experience

        self.reset(initial_exploration)

        self.old_ob = self.algo.env.reset()

    def sample(self, sess, itr):
        if self.adaptive and itr > 0 and self.batch_size < self.max_batch_size:
            if itr % self.batch_rate == 0:
                self.batch_size *= 2
        env = self.algo.env

        indices = random.sample(xrange(self.replay_size), self.batch_size)

        o = np.expand_dims(self.old_ob, 0)
        a, _ = self.algo.policy.sample_actions(o)
        o, r, done, _ = env.step(a[0, 0])
        self.store(self.old_ob, a, r, o, done)
        if done or itr % self.max_traj_len == 0:
            self.old_ob = env.reset()
        else:
            self.old_ob = o

        # convert samples to batch?
        samples = dict(observations=self._observations[indices], actions=self._actions[indices],
                       rewards=self._rewards[indices], terminals=self._terminals[indices],
                       next_observations=self._next_observations[indices])
        samples_info = [('r_ave', samples['rewards'].mean(), float)]
        return samples, samples_info

    def reset(self, initial_exploration):
        env = self.algo.env
        n_samples = 0
        traj_length = 0
        o = env.reset()
        while n_samples <= self.max_experience:
            a = env.action_space.sample()
            op, r, done, _ = env.step(a)
            self.store(o, a, r, op, done)
            o = op
            traj_length += 1
            n_samples += 1
            if traj_length > self.max_traj_len or done:
                o = env.reset()
                traj_length = 0
                continue

    def store(self, obs, action, reward, obsp, done):
        if self.replay_size < self.max_experience:
            self.replay_size += 1
        self._observations[self.head] = obs
        self._actions[self.head] = action
        self._rewards[self.head] = reward
        self._next_observations[self.head] = obsp
        self._terminals[self.head] = done
        self.head += 1
        if self.head >= self.max_experience:
            self.head = 0
