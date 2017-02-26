import hashlib
import json

import h5py
import numpy as np
import tensorflow as tf

from rltools import util, tfutil


class Model(object):

    def get_variables(self, trainable=False):
        """Get all or trainable variables in the graph"""
        if trainable:
            assert self.varscope
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.varscope.name)
        return tf.get_collection(tf.GraphKeys.VARIABLES, self.varscope.name)

    def get_num_params(self, trainable=False):
        return sum(v.get_shape().num_elements() for v in self.get_variables(trainable=trainable))

    @staticmethod
    def _hash_name2array(name2array):

        def hash_array(a):
            return '%.10f,%.10f,%d' % (np.mean(a), np.var(a), np.argmax(a))

        return hashlib.sha1('|'.join('%s %s'
                                     for n, h in sorted([(name, hash_array(a)) for name, a in
                                                         name2array]))).hexdigest()

    def savehash(self, sess):
        """Hash is based on values of variables"""
        vars_ = self.get_variables()
        vals = sess.run(vars_)
        return self._hash_name2array([(v.name, val) for v, val in util.safezip(vars_, vals)])

    # HDF5 saving and loading
    # The hierarchy in the HDF5 file reflects the hierarchy in the Tensorflow graph.
    def save_h5(self, sess, h5file, key, extra_attrs=None):
        with h5py.File(h5file, 'a') as f:
            if key in f:
                util.warn('WARNING: key {} already exists in {}'.format(key, h5file))
                dset = f[key]
            else:
                dset = f.create_group(key)

            vs = self.get_variables()
            vals = sess.run(vs)

            for v, val in util.safezip(vs, vals):
                dset[v.name] = val

            dset[self.varscope.name].attrs['hash'] = self.savehash(sess)
            if extra_attrs is not None:
                for k, v in extra_attrs:
                    if k in dset.attrs:
                        util.warn('Warning: attribute {} already exists in {}'.format(k, dset.name))
                    dset.attrs[k] = v

    def load_h5(self, sess, h5file, key):
        with h5py.File(h5file, 'r') as f:
            dset = f[key]

            ops = []
            for v in self.get_variables():
                util.header('Reading {}'.format(v.name))
                if v.name in dset:
                    ops.append(v.assign(dset[v.name][...]))
                else:
                    raise RuntimeError('Variable {} not found in {}'.format(v.name, dset))

            sess.run(ops)

            h = self.savehash(sess)
            try:
                assert h == dset[self.varscope.name].attrs[
                    'hash'], 'Checkpoint hash {} does not match loaded hash {}'.format(
                        dset[self.varscope.name].attrs['hash'], h)
            except AssertionError as err:
                util.warn('Checkpoint hash {} does not match loaded hash {}'.format(dset[
                    self.varscope.name].attrs['hash'], h))

# Layers for feedforward networks


class Layer(Model):

    @property
    def output(self):
        raise NotImplementedError

    @property
    def output_shape(self):
        """Shape refers to the shape without the batch axis, which always implicitly goes first"""
        raise NotImplementedError


class ReshapeLayer(Layer):

    def __init__(self, input_, new_shape, debug=False):
        self._output_shape = tuple(new_shape)
        if debug:
            util.header('Reshape(new_shape=%s)' % (str(self._output_shape),))
        with tf.variable_scope(type(self).__name__) as self.varscope:
            self._output = tf.reshape(input_, (-1,) + self._output_shape)

    @property
    def output(self):
        return self._output

    @property
    def output_shape(self):
        return self._output_shape


class FlattenLayer(Layer):
    """
    outdim: number of output dimensions - (B, N) by default so 2. For images set 3
    """

    def __init__(self, input_, outdim=2, debug=False):
        assert outdim >= 1
        self._outdim = outdim
        input_shape = tuple(input_.get_shape().as_list())
        to_flatten = input_shape[self._outdim - 1:]
        if any(s is None for s in to_flatten):
            flattened = None
        else:
            flattened = int(np.prod(to_flatten))

        self._output_shape = input_shape[1:self._outdim - 1] + (flattened,)
        if debug:
            util.header('Flatten(new_shape=%s)' % str(self._output_shape))
        pre_shape = tf.shape(input_)[:self._outdim - 1:]
        to_flatten = tf.reduce_prod(tf.shape(input_)[self._outdim - 1:])
        self._output = tf.reshape(input_, tf.concat(0, [pre_shape, tf.pack([to_flatten])]))

    @property
    def output(self):
        return self._output

    @property
    def output_shape(self):
        return self._output_shape


class AffineLayer(Layer):

    def __init__(self, input_B_Di, input_shape, output_shape, Winitializer, binitializer,
                 debug=False):
        assert len(input_shape) == len(output_shape) == 1
        if debug:
            util.header('Affine(in=%d, out=%d)' % (input_shape[0], output_shape[0]))
        self._output_shape = (output_shape[0],)
        with tf.variable_scope(type(self).__name__) as self.varscope:
            if Winitializer is None:
                Winitializer = tf.contrib.layers.xavier_initializer()
            if binitializer is None:
                binitializer = tf.zeros_initializer
            self.W_Di_Do = tf.get_variable('W', shape=[input_shape[0], output_shape[0]],
                                           initializer=Winitializer)
            self.b_1_Do = tf.get_variable('b', shape=[1, output_shape[0]], initializer=binitializer)
            self.output_B_Do = tf.matmul(input_B_Di, self.W_Di_Do) + self.b_1_Do

    @property
    def output(self):
        return self.output_B_Do

    @property
    def output_shape(self):
        return self._output_shape


class NonlinearityLayer(Layer):

    def __init__(self, input_B_Di, output_shape, func, debug=False):
        if debug:
            util.header('Nonlinearity(func=%s)' % func)
        self._output_shape = output_shape
        with tf.variable_scope(type(self).__name__) as self.varscope:
            self.output_B_Do = {'relu': tf.nn.relu,
                                'elu': tf.nn.elu,
                                'tanh': tf.tanh}[func](input_B_Di)

    @property
    def output(self):
        return self.output_B_Do

    @property
    def output_shape(self):
        return self._output_shape


class ConvLayer(Layer):

    def __init__(self, input_B_Ih_Iw_Ci, input_shape, Co, Fh, Fw, Sh, Sw, padding, initializer):
        assert len(input_shape) == 3
        Ih, Iw, Ci = input_shape
        if padding == 'SAME':
            Oh = np.ceil(float(Ih) / float(Sh))
            Ow = np.ceil(float(Iw) / float(Sw))
        elif padding == 'VALID':
            Oh = np.ceil(float(Ih - Fh + 1) / float(Sh))
            Ow = np.ceil(float(Iw - Fw + 1) / float(Sw))
        util.header(
            'Conv(chanin=%d, chanout=%d, filth=%d, filtw=%d, outh=%d, outw=%d, strideh=%d, stridew=%d, padding=%s)'
            % (Ci, Co, Fh, Fw, Oh, Ow, Sh, Sw, padding))
        self._output_shape = (Oh, Ow, Co)
        with tf.variable_scope(type(self).__name__) as self.varscope:
            if initializer is None:
                initializer = tf.contrib.layers.xavier_initializer()
            self.W_Fh_Fw_Ci_Co = tf.get_variable('W', shape=[Fh, Fw, Ci, Co],
                                                 initializer=initializer)
            self.b_1_1_1_Co = tf.get_variable('b', shape=[1, 1, 1, Co],
                                              initializer=tf.constant_initializer(0.))
            self.output_B_Oh_Ow_Co = tf.nn.conv2d(input_B_Ih_Iw_Ci, self.W_Fh_Fw_Ci_Co,
                                                  [1, Sh, Sw, 1], padding) + self.b_1_1_1_Co

    @property
    def output(self):
        return self.output_B_Oh_Ow_Co

    @property
    def output_shape(self):
        return self._output_shape

    @property
    def output(self):
        return self.output_B_Oh_Ow_Co

    @property
    def output_shape(self):
        return self._output_shape


class GRULayer(Layer):

    def __init__(self, input_B_T_Di, input_shape, hidden_units, hidden_nonlin, initializer,
                 hidden_init_trainable):
        if hidden_nonlin is None:
            hidden_nonlin = tf.identity

        self._hidden_units = hidden_units
        self._input_B_T_Di = input_B_T_Di
        self._input_shape = input_shape
        self.gate_nonlin = tf.nn.sigmoid
        self.hidden_nonlin = hidden_nonlin
        with tf.variable_scope(type(self).__name__) as self.varscope:
            if initializer is None:
                initializer = tf.contrib.layers.xavier_initializer()

            input_shape = self._input_shape  # (B, steps) removed
            input_dim = np.prod(input_shape)
            # Initial Hidden state weights
            self.h0 = tf.get_variable('h0', shape=[hidden_units], initializer=tf.zeros_initializer,
                                      trainable=hidden_init_trainable)

            with tf.variable_scope('reset'):
                # Reset Gate
                self.W_xr_Di_T = tf.get_variable('W_xr', shape=[input_dim, hidden_units],
                                                 initializer=initializer)
                self.W_hrT_T = tf.get_variable('W_hr', shape=[hidden_units, hidden_units],
                                               initializer=initializer)
                self.b_r_T = tf.get_variable('b_r', shape=[hidden_units],
                                             initializer=tf.constant_initializer(1.))

            with tf.variable_scope('update'):
                # Update Gate
                self.W_xu_Di_T = tf.get_variable('W_xu', shape=[input_dim, hidden_units],
                                                 initializer=initializer)
                self.W_huT_T = tf.get_variable('W_hu', shape=[hidden_units, hidden_units],
                                               initializer=initializer)
                self.b_u_T = tf.get_variable('b_u', shape=[hidden_units],
                                             initializer=tf.constant_initializer(1.))

            with tf.variable_scope('cell'):
                # Cell Gate
                self.W_xc_Di_T = tf.get_variable('W_xc', shape=[input_dim, hidden_units],
                                                 initializer=initializer)
                self.W_hcT_T = tf.get_variable('W_hc', shape=[hidden_units, hidden_units],
                                               initializer=initializer)
                self.b_c_T = tf.get_variable('b_c', shape=[hidden_units],
                                             initializer=tf.constant_initializer(0.))

            self.W_x_ruc_Di_3T = tf.concat(1, [self.W_xr_Di_T, self.W_xu_Di_T, self.W_xc_Di_T])
            self.W_h_ruc_T_3T = tf.concat(1, [self.W_hrT_T, self.W_huT_T, self.W_hcT_T])

        self._output_shape = (self._hidden_units,)

    def step(self, hprev, x):
        x_ruc = tf.matmul(x, self.W_x_ruc_Di_3T)
        h_ruc = tf.matmul(hprev, self.W_h_ruc_T_3T)
        x_r_Di_T, x_u_Di_T, x_c_Di_T = tf.split(split_dim=1, num_split=3, value=x_ruc)
        h_r, h_u, h_c = tf.split(split_dim=1, num_split=3, value=h_ruc)
        r = self.gate_nonlin(x_r_Di_T + h_r + self.b_r_T)
        u = self.gate_nonlin(x_u_Di_T + h_u + self.b_u_T)
        c = self.hidden_nonlin(x_c_Di_T + r * h_c + self.b_c_T)
        h = u * hprev + (1 - u) * c
        return h

    def step_layer(self, inp, prev_hidden):
        return GRUStepLayer([inp, prev_hidden], gru_layer=self)

    @property
    def output(self):
        """Iterate through hidden states to get outputs for all"""
        input_shape = tf.shape(self._input_B_T_Di)
        input = tf.reshape(self._input_B_T_Di, tf.pack([input_shape[0], input_shape[1], -1]))
        h0s = tf.tile(tf.reshape(self.h0, (1, self._hidden_units)), (input_shape[0], 1))
        # Flatten extra dimension
        shuffled_input = tf.transpose(input, (1, 0, 2))
        hs = tf.scan(self.step, elems=shuffled_input, initializer=h0s)
        shuffled_hs = tf.transpose(hs, (1, 0, 2))
        return shuffled_hs

    @property
    def output_shape(self):
        return self._output_shape


class GRUStepLayer(Layer):

    def __init__(self, inputs, gru_layer):
        assert all([not isinstance(inp, Layer) for inp in inputs])
        self.inputs = inputs
        self._gru_layer = gru_layer

    @property
    def output(self):
        x, hprev = self.inputs
        n_batch = tf.shape(x)[0]
        x = tf.reshape(x, tf.pack([n_batch, -1]))
        return self._gru_layer.step(hprev, x)

    @property
    def output_shape(self):
        return (self._gru_layer._hidden_units,)


def _check_keys(d, keys, optional):
    s = set(d.keys())
    if not (s == set(keys) or s == set(keys + optional)):
        raise RuntimeError('Got keys %s, but expected keys %s with optional keys %s' %
                           (str(s), str(keys), str(optional)))


def _parse_initializer(layerspec):
    if 'initializer' not in layerspec:
        return None
    initspec = layerspec['initializer']
    raise NotImplementedError('Unknown layer initializer type %s' % initspec['type'])


class FeedforwardNet(Layer):

    def __init__(self, input_B_Di, input_shape, layerspec_json, debug=False):
        """
        Args:
            layerspec (string): JSON string describing layers
        """
        assert len(input_shape) >= 1
        self.input_B_Di = input_B_Di

        layerspec = json.loads(layerspec_json)
        if debug:
            util.ok('Loading feedforward net specification')
            util.header(json.dumps(layerspec, indent=2, separators=(',', ': ')))

        self.layers = []
        with tf.variable_scope(type(self).__name__) as self.varscope:

            prev_output, prev_output_shape = input_B_Di, input_shape

            for i_layer, ls in enumerate(layerspec):
                with tf.variable_scope('layer_%d' % i_layer):
                    if ls['type'] == 'reshape':
                        _check_keys(ls, ['type', 'new_shape'], [])
                        self.layers.append(ReshapeLayer(prev_output, ls['new_shape'], debug=debug))

                    elif ls['type'] == 'flatten':
                        _check_keys(ls, ['type'], [])
                        self.layers.append(FlattenLayer(prev_output, debug=debug))

                    elif ls['type'] == 'fc':
                        _check_keys(ls, ['type', 'n'], ['initializer'])
                        self.layers.append(
                            AffineLayer(prev_output, prev_output_shape, output_shape=(ls['n'],),
                                        Winitializer=_parse_initializer(ls), binitializer=None,
                                        debug=debug))

                    elif ls['type'] == 'conv':
                        _check_keys(ls, ['type', 'chanout', 'filtsize', 'stride', 'padding'],
                                    ['initializer'])
                        self.layers.append(
                            ConvLayer(input_B_Ih_Iw_Ci=prev_output, input_shape=prev_output_shape,
                                      Co=ls['chanout'], Fh=ls['filtsize'], Fw=ls['filtsize'], Sh=ls[
                                          'stride'], Sw=ls['stride'], padding=ls['padding'],
                                      initializer=_parse_initializer(ls)))

                    elif ls['type'] == 'nonlin':
                        _check_keys(ls, ['type', 'func'], [])
                        self.layers.append(
                            NonlinearityLayer(prev_output, prev_output_shape, ls['func'],
                                              debug=debug))

                    else:
                        raise NotImplementedError('Unknown layer type %s' % ls['type'])

                prev_output, prev_output_shape = self.layers[-1].output, self.layers[
                    -1].output_shape
                self._output, self._output_shape = prev_output, prev_output_shape

    @property
    def output(self):
        return self._output

    @property
    def output_shape(self):
        return self._output_shape


class GRUNet(Layer):
    # Mostly based on rllab's
    def __init__(self,
                 input_B_T_Di,
                 input_shape,
                 output_dim,
                 layer_specjson,  # hidden_dim, output_dim, hidden_nonlin=tf.nn.relu,
                 # hidden_init_trainable=False
                 debug=False):
        layerspec = json.loads(layer_specjson)
        if debug:
            util.ok('Loading GRUNet specification')
            util.header(json.dumps(layerspec, indent=2, separators=(',', ': ')))
        self._hidden_dim = layerspec['gru_hidden_dim']
        self._hidden_nonlin = {'relu': tf.nn.relu,
                               'elu': tf.nn.elu,
                               'tanh': tf.tanh,
                               'identity': tf.identity}[layerspec['gru_hidden_nonlin']]
        self._hidden_init_trainable = layerspec['gru_hidden_init_trainable']
        self._output_dim = output_dim
        assert len(input_shape) >= 1  # input_shape is Di
        self.input_B_T_Di = input_B_T_Di
        with tf.variable_scope(type(self).__name__) as self.varscope:
            if 'feature_net' in layerspec:
                _feature_net = FeedforwardNet(input_B_T_Di, input_shape, layerspec['feature_net'])
                self._feature_shape = _feature_net.output_shape
                self._feature = tf.reshape(_feature_net.output,
                                           tf.pack([tf.shape(self.input_B_T_Di)[0],
                                                    tf.shape(self.input_B_T_Di)[1],
                                                    self._feature_shape[-1]]))
            else:
                self._feature_shape = input_shape
                self._feature = input_B_T_Di
            self._step_input = tf.placeholder(tf.float32, shape=(None,) + self._feature_shape,
                                              name='step_input')
            self._step_prev_hidden = tf.placeholder(tf.float32, shape=(None, self._hidden_dim),
                                                    name='step_prev_hidden')

            self._gru_layer = GRULayer(self._feature, self._feature_shape,
                                       hidden_units=self._hidden_dim,
                                       hidden_nonlin=self._hidden_nonlin, initializer=None,
                                       hidden_init_trainable=self._hidden_init_trainable)
            self._gru_flat_layer = ReshapeLayer(self._gru_layer.output,
                                                (self._hidden_dim,))  # (B*step, hidden_dim)
            self._output_flat_layer = AffineLayer(self._gru_flat_layer.output,
                                                  self._gru_flat_layer.output_shape,
                                                  output_shape=(self._output_dim,),
                                                  Winitializer=None, binitializer=None)

            self._output = tf.reshape(self._output_flat_layer.output, tf.pack(
                (tf.shape(self.input_B_T_Di)[0], tf.shape(self.input_B_T_Di)[1], -1)))
            self._output_shape = (self._output_flat_layer.output_shape[-1],)
            self._step_hidden_layer = self._gru_layer.step_layer(self._step_input,
                                                                 self._step_prev_hidden)
            self._step_output = tf.matmul(
                self._step_hidden_layer.output,
                self._output_flat_layer.W_Di_Do) + self._output_flat_layer.b_1_Do
            self._hid_init = self._gru_layer.h0

    @property
    def output(self):
        return self._output

    @property
    def output_shape(self):
        return self._output_shape

    @property
    def step_input(self):
        return self._step_input

    @property
    def step_hidden(self):
        return self._step_hidden_layer.output

    @property
    def step_prev_hidden(self):
        return self._step_prev_hidden

    @property
    def step_output(self):
        return self._step_output

    @property
    def hid_init(self):
        return self._hid_init


class NoOpStandardizer(object):

    def __init__(self, dim, eps=1e-6):
        pass

    def update(self, points_N_D, **kwargs):
        pass

    def standardize_expr(self, x_B_D):
        return x_B_D

    def unstandardize_expr(self, y_B_D):
        return y_B_D

    def standardize(self, x_B_D, **kwargs):
        return x_B_D

    def unstandardize(self, y_B_D, **kwargs):
        return y_B_D


class Standardizer(Model):

    def __init__(self, shape, eps=1e-6, init_count=0, init_mean=0., init_meansq=1.):
        """
        Args:
            shape: dimension of the space of points to be standardized
            eps: small constant to add to denominators to prevent division by 0
            init_count, init_mean, init_meansq: initial values for accumulators

        Note:
            if init_count is 0, then init_mean and init_meansq have no effect beyond
            the first call to update(), which will ignore their values and
            replace them with values from a new batch of data.
        """
        self._eps = eps
        if isinstance(shape, int):
            shape = (shape,)
        self._shape = shape
        with tf.variable_scope(type(self).__name__) as self.varscope:
            self._count = tf.get_variable('count', shape=(1,),
                                          initializer=tf.constant_initializer(init_count),
                                          trainable=False)
            self._mean_1_D = tf.get_variable('mean_1_D', shape=(1,) + self._shape,
                                             initializer=tf.constant_initializer(init_mean),
                                             trainable=False)
            self._meansq_1_D = tf.get_variable('meansq_1_D', shape=(1,) + self._shape,
                                               initializer=tf.constant_initializer(init_meansq),
                                               trainable=False)
            self._stdev_1_D = tf.sqrt(self._meansq_1_D - tf.square(self._mean_1_D) + self._eps)

        self.get_mean = tfutil.function([], self._mean_1_D)
        self.get_meansq = tfutil.function([], self._meansq_1_D)
        self.get_stdev = tfutil.function([], self._stdev_1_D)
        self.get_count = tfutil.function([], self._count)

    def update(self, points_N_D, **kwargs):
        assert points_N_D.ndim >= 2 and points_N_D.shape[1:] == self._shape
        num = points_N_D.shape[0]
        count = self.get_count(**kwargs)
        a = count / (count + num)
        mean_op = self._mean_1_D.assign(a * self.get_mean(**kwargs) + (1. - a) * points_N_D.mean(
            axis=0, keepdims=True))
        meansq_op = self._meansq_1_D.assign(a * self.get_meansq(**kwargs) + (1. - a) * (
            points_N_D**2).mean(axis=0, keepdims=True))
        count_op = self._count.assign(count + num)
        sess = kwargs.pop('sess', tf.get_default_session())
        sess.run([mean_op, meansq_op, count_op])

    def standardize_expr(self, x_B_D):
        return (x_B_D - self._mean_1_D) / (self._stdev_1_D + self._eps)

    def unstandardize_expr(self, y_B_D):
        return y_B_D * (self._stdev_1_D + self._eps) + self._mean_1_D

    def standardize(self, x_B_D, centered=True, **kwargs):
        assert x_B_D.ndim >= 2
        mu = 0.
        if centered:
            mu = self.get_mean(**kwargs)
        return (x_B_D - mu) / (self.get_stdev(**kwargs) + self._eps)

    def unstandardize(self, y_B_D, centered=True, **kwargs):
        assert y_B_D.ndim >= 2
        mu = 0.
        if centered:
            mu = self.get_mean(**kwargs)
        return y_B_D * (self.get_stdev(**kwargs) + self._eps) + mu
