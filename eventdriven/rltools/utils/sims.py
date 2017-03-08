import numpy as np

def simulate(env, policy, maxsteps, render=False):
    obs = env.reset()
    rtot = 0.0
    for _ in xrange(maxsteps):
        if render: env.render()
        action, action_dist = policy.act(obs)
        obs, reward, done, _ = env.step(action)
        rtot += reward
        if done:
            break
    return rtot



def evaluate(env, policy, maxsteps, n_traj, render=False):
    rewards = np.zeros(n_traj)
    for i in xrange(n_traj):
        obs = env.reset()
        rtot = 0.0
        for _ in xrange(maxsteps):
            if render: env.render()
            action, action_dist = policy.act(obs)
            obs, reward, done, _ = env.step(action)
            rtot += reward
            if done:
                break
        rewards[i] = rtot
    return rewards


def evaluate_time(env, policy, maxsteps, n_traj):
    rewards = np.zeros(n_traj)
    times = np.zeros(n_traj)
    for i in xrange(n_traj):
        obs = env.reset()
        t = 0
        rtot = 0.0
        for _ in xrange(maxsteps):
            action, action_dist = policy.act(obs)
            obs, reward, done, _ = env.step(action)
            t += 1
            rtot += reward
            if done:
                break
        times[i] = t
        rewards[i] = rtot
    return rewards, times
