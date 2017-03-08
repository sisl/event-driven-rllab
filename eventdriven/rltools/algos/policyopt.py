from __future__ import absolute_import, print_function
import numpy as np

from rltools import util, trajutil
from rltools.algos import RLAlgorithm
from rltools.samplers.serial import SimpleSampler
from rltools.samplers.parallel import ParallelSampler


class SamplingPolicyOptimizer(RLAlgorithm):

    def __init__(self, env, policy, baseline, step_func, discount=0.99, gae_lambda=1, n_iter=500,
                 start_iter=0, center_adv=True, positive_adv=False, store_paths=False,
                 whole_paths=True, sampler_cls=None,
                 sampler_args=dict(max_traj_len=200, n_timesteps=6400, adaptive=False,
                                   n_timesteps_min=1600, n_timesteps_max=12800, timestep_rate=40,
                                   enable_rewnorm=True), update_curriculum=False, **kwargs):
        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.step_func = step_func
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.n_iter = n_iter
        self.start_iter = start_iter
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths  # TODO
        self.whole_paths = whole_paths  # TODO
        if sampler_cls is None:
            sampler_cls = SimpleSampler
        self.sampler = sampler_cls(self, **sampler_args)
        self.update_curriculum = update_curriculum
        self.total_time = 0.0

    def train(self, sess, log, save_freq, **kwargs):
        for itr in range(self.start_iter, self.n_iter):
            iter_info = self.step(sess, itr)
            log.write(iter_info, print_header=itr % 20 == 0)
            if itr % save_freq == 0 or itr % self.n_iter:
                log.write_snapshot(sess, self.policy, itr)
            if self.update_curriculum:
                self.env.update_curriculum(itr)

    def step(self, sess, itr):
        with util.Timer() as t_all:
            # Sample trajs using current policy
            with util.Timer() as t_sample:
                if itr == 0:
                    # extra batch to init std
                    trajbatch0, _ = self.sampler.sample(sess, itr)
                    self.policy.update_obsnorm(trajbatch0.obs.stacked, sess=sess)
                    self.baseline.update_obsnorm(trajbatch0.obs.stacked, sess=sess)
                    self.sampler.rewnorm.update(trajbatch0.r.stacked[:, None], sess=sess)
                trajbatch, sample_info_fields = self.sampler.sample(sess, itr)

            # Compute baseline
            with util.Timer() as t_base:
                trajbatch_vals, base_info_fields = self.sampler.process(sess, itr, trajbatch,
                                                                        self.discount,
                                                                        self.gae_lambda,
                                                                        self.baseline)

            # Take the policy grad step
            with util.Timer() as t_step:
                params0_P = self.policy.get_params(sess=sess)
                step_print_fields = self.step_func(sess, self.policy, trajbatch,
                                                   trajbatch_vals['advantage'])
                self.policy.update_obsnorm(trajbatch.obs.stacked, sess=sess)
                self.sampler.rewnorm.update(trajbatch.r.stacked[:, None], sess=sess)
        # LOG
        self.total_time += t_all.dt

        fields = [
            ('iter', itr, int)
        ] + sample_info_fields + [
            ('vf_r2', trajbatch_vals['v_r'], float),
            ('tdv_r2', trajbatch_vals['tv_r'], float),
            ('ent', self.policy._compute_actiondist_entropy(trajbatch.adist.stacked).mean(), float
            ),  # entropy of action distribution
            ('dx', util.maxnorm(params0_P - self.policy.get_params()), float
            )  # max parameter different from last iteration
        ] + base_info_fields + step_print_fields + [
            ('tsamp', t_sample.dt, float),  # Time for sampling
            ('tbase', t_base.dt, float),  # Time for advantage/baseline computation
            ('tstep', t_step.dt, float),
            ('ttotal', self.total_time, float)
        ]
        return fields


def TRPO(max_kl, subsample_hvp_frac=.1, damping=1e-2, grad_stop_tol=1e-6, max_cg_iter=10,
         enable_bt=True):

    def trpo_step(sess, policy, trajbatch, advantages):

        if policy.recurrent:
            # standardize advantage
            advpadded_N_H = (advantages.padded(fill=0.) - np.mean(advantages.stacked)) / (
                np.std(advantages.stacked) + 1e-8)
            valid = trajutil.RaggedArray([np.ones(trajlen) for trajlen in advantages.lengths])
            feed = (trajbatch.obs.padded(fill=0.), trajbatch.a.padded(fill=0.),
                    trajbatch.adist.padded(fill=0.), advpadded_N_H, valid.padded(fill=0.))
        else:
            # standardize advantage
            advstacked_N = util.standardized(advantages.stacked)
            # Compute objective, KL divergence and gradietns at init point
            feed = (trajbatch.obs.stacked, trajbatch.a.stacked, trajbatch.adist.stacked,
                    advstacked_N)

        step_info = policy._ngstep(sess, feed, max_kl=max_kl, damping=damping,
                                   subsample_hvp_frac=subsample_hvp_frac,
                                   grad_stop_tol=grad_stop_tol)
        return [
            ('dl', step_info.obj1 - step_info.obj0, float),  # Improvement in objective
            ('kl', step_info.kl1, float),  # kl cost
            ('gnorm', step_info.gnorm, float),  # gradient norm
            ('bt', step_info.bt, int),  # backtracking steps
        ]

    return trpo_step


class ConcurrentPolicyOptimizer(RLAlgorithm):

    def __init__(self, env, policies, baselines, step_func, discount, gae_lambda, n_iter,
                 start_iter=0, sampler_cls=None, sampler_args=None, target_policy=None,
                 interp_alpha=None, **kwargs):
        self.env = env
        self.policies = policies
        self.baselines = baselines
        self.step_func = step_func
        self.target_policy = target_policy
        self.interp_alpha = interp_alpha
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.n_iter = n_iter
        self.start_iter = start_iter
        if sampler_cls is None:
            sampler_cls = ParallelSampler
        self.sampler = sampler_cls(self, **sampler_args)
        self.total_time = 0.0

    def train(self, sess, log, save_freq, blend_freq=0, keep_kmax=0, blend_eval_trajs=50):
        for itr in range(self.start_iter, self.n_iter):
            iter_info = self.step(sess, itr)
            log.write(iter_info, print_header=itr % 20 == 0)
            if itr % save_freq == 0 or itr % self.n_iter:
                for policy in self.policies:
                    log.write_snapshot(sess, policy, itr)

            if blend_freq > 0:
                # Blending does not work
                assert self.target_policy is not None
                if itr == 0:
                    params_P_ag = [policy.get_params() for policy in self.policies]
                    weights, evalrewards = self._eval_policy_weights(blend_eval_trajs)
                    weightparams_P = np.sum([w * p for w, p in util.safezip(weights, params_P_ag)],
                                            axis=0)

                    blendparams_P = 0.001 * self.target_policy.get_params() + 0.999 * weightparams_P
                if itr > 0 and (itr % blend_freq == 0 or itr % self.n_iter == 0):
                    params_P_ag = [policy.get_params() for policy in self.policies]
                    weights, evalrewards = self._eval_policy_weights(blend_eval_trajs)
                    weightparams_P = np.sum([w * p for w, p in util.safezip(weights, params_P_ag)],
                                            axis=0)

                    blendparams_P = self.interp_alpha * self.target_policy.get_params() + (
                        1 - self.interp_alpha) * weightparams_P

                self.target_policy.set_params(blendparams_P)
                log.write_snapshot(sess, self.target_policy, itr)
                if keep_kmax:
                    keep_inds = np.argpartition(evalrewards, -keep_kmax)[-keep_kmax:]
                else:
                    keep_inds = []
                for agid, policies in enumerate(self.policies):
                    if agid in keep_inds:
                        continue
                    policies.set_params(blendparams_P)

    def _eval_policy_weights(self, eval_trajs):
        evalrewards = np.zeros(len(self.env.agents))
        n_workers = self.sampler.n_workers if hasattr(self.sampler, 'n_workers') else 4
        # Rewards when all agents have the same policy
        for agid, policy in enumerate(self.policies):
            evalrewards[agid] = np.mean(
                util.evaluate_policy(self.env, [
                    policy for _ in range(len(self.env.agents))
                ], n_trajs=eval_trajs, deterministic=False, max_traj_len=self.sampler.max_traj_len,
                                     mode='concurrent', disc=self.discount, n_workers=n_workers)[
                                         'ret'])

        weights = evalrewards / np.sum(evalrewards)
        if all(evalrewards < 0):
            weights = 1 - weights
        return weights, evalrewards

    def step(self, sess, itr):
        with util.Timer() as t_all:
            with util.Timer() as t_sample:
                if itr == 0:
                    # extra batch to init std
                    trajbatchlist0, _ = self.sampler.sample(sess, itr)
                    for policy, baseline, trajbatch0 in util.safezip(self.policies, self.baselines,
                                                                     trajbatchlist0):
                        policy.update_obsnorm(trajbatch0.obs.stacked, sess=sess)
                        baseline.update_obsnorm(trajbatch0.obs.stacked, sess=sess)
                        self.sampler.rewnorm.update(trajbatch0.r.stacked[:, None], sess=sess)
                trajbatchlist, sampler_info_fields = self.sampler.sample(sess, itr)

            # Baseline
            with util.Timer() as t_base:
                trajbatch_vals_list, base_info_fields_list = [], []
                for agid, trajbatch in enumerate(trajbatchlist):
                    trajbatch_vals, base_info_fields = self.sampler.process(sess, itr, trajbatch,
                                                                            self.discount,
                                                                            self.gae_lambda,
                                                                            self.baselines[agid])
                    trajbatch_vals_list.append(trajbatch_vals)
                    base_info_fields_list += base_info_fields

            # Take policy steps
            with util.Timer() as t_step:
                step_print_fields_list = []
                params0_P_list = []
                for agid, policy in enumerate(self.policies):
                    params0_P = policy.get_params()
                    params0_P_list.append(params0_P)
                    step_print_fields = self.step_func(sess, policy, trajbatchlist[agid],
                                                       trajbatch_vals_list[agid]['advantage'])
                    step_print_fields_list += step_print_fields
                    policy.update_obsnorm(trajbatchlist[agid].obs.stacked, sess=sess)
                    self.sampler.rewnorm.update(trajbatchlist[agid].r.stacked[:, None], sess=sess)

        # LOG
        self.total_time += t_all.dt

        infos = []
        for agid in range(len(self.policies)):
            infos += [
                ('vf_r2_{}'.format(agid), trajbatch_vals_list[agid]['v_r'], float),
                ('tdv_r2_{}'.format(agid), trajbatch_vals_list[agid]['tv_r'], float),
                ('ent_{}'.format(agid), self.policies[agid]._compute_actiondist_entropy(
                    trajbatchlist[agid].adist.stacked).mean(), float),
                ('dx_{}'.format(agid),
                 util.maxnorm(params0_P_list[agid] - self.policies[agid].get_params()), float)
            ]
        fields = [
            ('iter', itr, int)
        ] + sampler_info_fields + infos + base_info_fields_list + step_print_fields_list + [
            ('tsamp', t_sample.dt, float),  # Time for sampling
            ('tbase', t_base.dt, float),  # Time for advantage/baseline computation
            ('tstep', t_step.dt, float),
            ('ttotal', self.total_time, float)
        ]
        return fields
