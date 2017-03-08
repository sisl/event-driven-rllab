from __future__ import print_function

import errno
import os
import timeit

import h5py
import numpy as np
from colorama import Fore, Style


class Timer(object):

    def __enter__(self):
        self.t_start = timeit.default_timer()
        return self

    def __exit__(self, _1, _2, _3):
        self.t_end = timeit.default_timer()
        self.dt = self.t_end - self.t_start


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def split_h5_name(fullpath, sep='/'):
    """
    From h5ls.c:
     * Example: ../dir1/foo/bar/baz
     *          \_________/\______/
     *             file       obj
     *
    """
    sep_inds = [i for i, c in enumerate(fullpath) if c == sep]
    for sep_idx in sep_inds:
        filename, objname = fullpath[:sep_idx], fullpath[sep_idx:]
        if not filename:
            continue
        # Try to open the file. If it fails, try the next separation point.
        try:
            h5py.File(filename, 'r').close()
        except IOError:
            continue
        # It worked!
        return filename, objname
    raise IOError('Could not open HDF5 file/object {}'.format(fullpath))


def discount(r_N_T_D, gamma):
    '''
    Computes Q values from rewards.
    q_N_T_D[i,t,:] == r_N_T_D[i,t,:] + gamma*r_N_T_D[i,t+1,:] + gamma^2*r_N_T_D[i,t+2,:] + ...
    '''
    assert r_N_T_D.ndim == 2 or r_N_T_D.ndim == 3
    input_ndim = r_N_T_D.ndim
    if r_N_T_D.ndim == 2:
        r_N_T_D = r_N_T_D[..., None]

    discfactors_T = np.power(gamma, np.arange(r_N_T_D.shape[1]))
    discounted_N_T_D = r_N_T_D * discfactors_T[None, :, None]
    q_N_T_D = np.cumsum(
        discounted_N_T_D[:, ::-1, :],
        axis=1)[:, ::
                -1, :]  # this is equal to gamma**t * (r_N_T_D[i,t,:] + gamma*r_N_T_D[i,t+1,:] + ...)
    q_N_T_D /= discfactors_T[None, :, None]

    # Sanity check: Q values at last timestep should equal original rewards
    assert np.allclose(q_N_T_D[:, -1, :], r_N_T_D[:, -1, :])

    if input_ndim == 2:
        assert q_N_T_D.shape[-1] == 1
        return q_N_T_D[:, :, 0]
    return q_N_T_D


def standardized(a):
    out = a.copy()
    out -= a.mean()
    out /= a.std() + 1e-8
    return out


def safezip(*ls):
    assert all(len(l) == len(ls[0]) for l in ls)
    return zip(*ls)


def maxnorm(a):
    return np.abs(a).max()


def gather(vals, idx):
    return vals[idx]


def lookup_last_idx(a, inds):
    """
    Looks up indices in a. e.g. a[[1, 2, 3]] = [a[1], a[2], a[3]]
    a is a d1 x d2 ... dn array
    inds is a d1 x d2 ... d(n-1) array of integers
    returns the array
    out[i_1,...,i_{n-1}] = a[i_1,...,i_{n-1}, inds[i_1,...,i_{n-1}]]
    """

    # Flatten the arrays
    ashape, indsshape = np.shape(a), np.shape(inds)
    aflat, indsflat = np.reshape(a, (-1,)), np.reshape(inds, (-1,))

    # Compute the indices corresponding to inds in the flattened array
    delta = gather(ashape, np.size(ashape) - 1)  # i.e. delta = ashape[-1],
    aflatinds = np.arange(0, stop=np.size(a), step=delta) + indsflat

    # Look up the desired elements in the flattened array, and reshape
    # to the original shape
    return np.reshape(gather(aflat, aflatinds), indsshape)


def stack_dict_list(dict_list):
    ret = dict()
    if not dict_list:
        return ret
    keys = dict_list[0].keys()
    for k in keys:
        eg = dict_list[0][k]
        if isinstance(eg, dict):
            v = stack_dict_list([x[k] for x in dict_list])
        else:
            v = np.array([x[k] for x in dict_list])
        ret[k] = v

    return ret


def evaluate_policy(env, policy, n_trajs, deterministic, max_traj_len, mode, disc, n_workers=4):
    ok('Sampling {} trajs (max len {}) from policy in {}'.format(n_trajs, max_traj_len, env))

    # Sample
    from rltools.samplers.parallel import RolloutProxy
    from six.moves import cPickle
    import time
    from gevent import Timeout
    from rltools.trajutil import TrajBatch
    proxies = [RolloutProxy(env, policy, max_traj_len, mode, i, 0) for i in range(n_workers)]

    if mode == 'concurrent':
        state_str = cPickle.dumps([p.get_state() for p in policy])
    else:
        state_str = cPickle.dumps(policy.get_state(), protocol=-1)
    for proxy in proxies:
        proxy.client("set_state", state_str, async=True)

    seed_idx = 0
    seed_idx2 = seed_idx
    worker2job = {}

    def assign_job_to(i_worker, seed):
        worker2job[i_worker] = (seed, proxies[i_worker].client("sample", seed, async=True))
        seed += 1
        return seed

    # Start jobs
    for i_worker in range(n_workers):
        seed_idx2 = assign_job_to(i_worker, seed_idx2)

    trajs_so_far = 0
    seed2traj = {}
    while True:
        for i_worker in range(n_workers):
            try:
                (seed_idx, future) = worker2job[i_worker]
                traj_string = future.get(timeout=1e-3)  # XXX
            except Timeout:
                pass
            else:
                traj = cPickle.loads(traj_string)
                seed2traj[seed_idx] = traj
                trajs_so_far += 1
                if trajs_so_far >= n_trajs:
                    break
                else:
                    seed_idx2 = assign_job_to(i_worker, seed_idx2)
        if trajs_so_far >= n_trajs:
            break
        time.sleep(0.01)

    # Wait until all jobs finish
    for seed_idx, future in worker2job.values():
        seed2traj[seed_idx] = cPickle.loads(future.get())

    trajs = []

    for (seed, traj) in seed2traj.items():
        trajs.append(traj)
        trajs_so_far += 1

    # Trajs
    if mode == 'centralized':
        trajbatch = TrajBatch.FromTrajs(trajs)
        r_B_T = trajbatch.r.padded(fill=0.)
        ret = r_B_T.sum(axis=1).mean()
        discret = discount(r_B_T, disc).mean()
        info = {tinfo[0]: np.mean(tinfo[1]) for tinfo in trajbatch.info}
        return dict(ret=ret, disc_ret=discret, **info)
    elif mode in ['decentralized', 'concurrent']:
        agent2trajs = {}
        for agid in range(len(env.agents)):
            agent2trajs[agid] = []
        for envtrajs in trajs:
            for agid, agtraj in enumerate(envtrajs):
                agent2trajs[agid].append(agtraj)

        agent2trajbatch = {}
        rets, retsstd = [], []
        discrets = []
        infos = []
        for agent, trajs in agent2trajs.items():
            agent2trajbatch[agent] = TrajBatch.FromTrajs(trajs)
            r_B_T = agent2trajbatch[agent].r.padded(fill=0.)
            rets.append(r_B_T.sum(axis=1).mean())
            retsstd.append(r_B_T.sum(axis=1).std())
            discrets.append(discount(r_B_T, disc).mean())
            infos.append({tinfo[0]: np.mean(tinfo[1]) for tinfo in agent2trajbatch[agent].info})
        infos = stack_dict_list(infos)
        return dict(ret=rets, retstd=retsstd, disc_ret=discrets, **infos)
    else:
        raise NotImplementedError()


class Color(object):
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def header(s):
    print(Color.HEADER + '{}'.format(s) + Color.ENDC)


def warn(s):
    print(Color.WARNING + '{}'.format(s) + Color.ENDC)


def failure(s):
    print(Color.FAIL + '{}'.format(s) + Color.ENDC)


def ok(s):
    print(Color.OKBLUE + '{}'.format(s) + Color.ENDC)


class EzPickle(object):
    """Objects that are pickled and unpickled via their constructor
    arguments.

    Example usage:

        class Dog(Animal, EzPickle):
            def __init__(self, furcolor, tailkind="bushy"):
                Animal.__init__()
                EzPickle.__init__(furcolor, tailkind)
                ...

    When this object is unpickled, a new Dog will be constructed by passing the provided
    furcolor and tailkind into the constructor. However, philosophers are still not sure
    whether it is still the same dog.

    """

    def __init__(self, *args, **kwargs):
        self._ezpickle_args = args
        self._ezpickle_kwargs = kwargs

    def __getstate__(self):
        return {"_ezpickle_args": self._ezpickle_args, "_ezpickle_kwargs": self._ezpickle_kwargs}

    def __setstate__(self, d):
        out = type(self)(*d["_ezpickle_args"], **d["_ezpickle_kwargs"])
        self.__dict__.update(out.__dict__)
