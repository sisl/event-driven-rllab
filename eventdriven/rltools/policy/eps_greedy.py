from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf
import random

from rltools import nn, tfutil
from rltools.distributions import Categorical, RecurrentCategorical
from rltools.policy.stochastic import StochasticPolicy
from rltools.policy.value import ValuePolicy


class EpsGreedyMLPPolicy(ValuePolicy):

    def __init__(self, obsfeat_space, action_space, hidden_spec, enable_obsnorm, tblog,
                 varscope_name, eps=1.0):
        self.hidden_spec = hidden_spec
        self.eps = eps
        super(EpsGreedyMLPPolicy, self).__init__(obsfeat_space, action_space, action_space.n,
                                                   enable_obsnorm, tblog, varscope_name)


    def _make_actionval_ops(self, obsfeat_B_Df):
        with tf.variable_scope('hidden'):
            net = nn.FeedforwardNet(obsfeat_B_Df, self.obsfeat_space.shape, self.hidden_spec)
        with tf.variable_scope('out'):
            out_layer = nn.AffineLayer(net.output, net.output_shape, (self.action_space.n,),
                                       initializer=tf.zeros_initializer)  # TODO action_space
        vals_B_Pa = out_layer.output
        return vals_B_Pa

    def _sample_from_actiondist(self, actionvals_B_Pa, deterministic=False):                                                            
        assert actionvals_B_Pa.shape[1] == self.action_space.n
        if deterministic: return np.argmax(actionvals_B_Pa, axis=1)[:, None]
        if random.random() < self.eps: # TODO: should pass eps somehow?
            return np.expand_dims([self.action_space.sample()],0)
        else:
            return np.argmax(actionvals_B_Pa, axis=1)[:, None]

    def set_eps(self, eps):
        self.eps = eps

