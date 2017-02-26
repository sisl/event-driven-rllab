from rltools import nn


class Policy(nn.Model):

    def __init__(self, observation_space, action_space):
        self._observation_space = observation_space
        self._action_space = action_space

    def reset(self, *args, **kwargs):
        pass

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def recurrent(self):
        """Indicate whether the policy is recurrent"""
        return False
