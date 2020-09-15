from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras.engine import InputSpec
from keras.engine.topology import Layer

import tensorflow as tf


class DynamicPaddingLayer(Layer):
    '''Adds padding to input tensor such that its spatial dimensions
    are divisible by a given factor.

    '''
    def __init__(self, factor, ndim=4, data_format=None, **kwargs):
        '''
        '''
        if data_format is None:
            data_format = K.image_data_format()
        assert data_format in {'channels_last', 'channels_first'}

        self.ndim = ndim
        self.data_format = data_format
        self.input_spec = [InputSpec(ndim=self.ndim)]
        self.factor = factor
        super(DynamicPaddingLayer, self).__init__(**kwargs)

    def get_padded_dim(self, size):
        '''
        '''
        if size is None:
            return size
        if size % self.factor == 0:
            return size
        return size + self.factor - size % self.factor

    def get_paddings(self, size):
        '''
        '''
        dx = self.factor - size % self.factor
        return [dx // 2, dx - dx // 2]

    def compute_output_shape(self, input_shape):
        '''
        '''
        ndim = len(input_shape)
        if self.data_format == 'channels_last':
            return (input_shape[0], ) + tuple(
                self.get_padded_dim(input_shape[dim])
                for dim in range(1, ndim - 1)) + (input_shape[-1], )

        return (input_shape[0], input_shape[1]) + tuple(
            self.get_padded_dim(input_shape[dim]) for dim in range(2, ndim))

    def call(self, inputs):
        '''
        '''
        input_shape = tf.shape(inputs)
        ndim = K.ndim(inputs)
        if self.data_format == 'channels_last':
            paddings = [[0, 0]] + [
                self.get_paddings(input_shape[dim])
                for dim in range(1, ndim - 1)
            ] + [[0, 0]]
        else:
            paddings = [[0, 0], [0, 0]] + [
                self.get_paddings(input_shape[dim]) for dim in range(2, ndim)
            ]

        return tf.pad(inputs, paddings, 'CONSTANT')

    def get_config(self):
        config = {
            'factor': self.factor,
            'data_format': self.data_format,
            'ndim': self.ndim
        }
        base_config = super(DynamicPaddingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DynamicTrimmingLayer(Layer):
    '''Trims a given tensor to have the same spatial dimensions
    as another.
    '''
    def __init__(self, ndim=4, data_format=None, **kwargs):
        '''
        '''

        self.ndim = ndim
        if data_format is None:
            data_format = K.image_data_format()
        assert data_format in {'channels_last', 'channels_first'}
        self.data_format = data_format

        self.input_spec = [
            InputSpec(ndim=self.ndim),
            InputSpec(ndim=self.ndim)
        ]
        super(DynamicTrimmingLayer, self).__init__(**kwargs)

    def get_config(self):
        config = {'data_format': self.data_format, 'ndim': self.ndim}
        base_config = super(DynamicTrimmingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        '''
        '''
        if self.data_format == 'channels_last':
            return input_shape[0][:-1] + input_shape[1][-1:]
        return input_shape[1][:2] + input_shape[0][2:]

    def call(self, inputs):
        '''expects a list of exactly 2 inputs:

        inputs[0] = original_tensor -> for shape
        inputs[1] = output tensor that should be trimmed

        '''
        assert len(inputs) == 2
        output_tensor = inputs[1]
        original_shape = tf.shape(inputs[0])
        output_shape = tf.shape(output_tensor)

        dx = [(x - y) // 2
              for x, y in ((output_shape[idx], original_shape[idx])
                           for idx in range(self.ndim))]

        if self.data_format == 'channels_last':
            starts = [0] + dx[1:-1] + [0]
            ends = [-1] + [
                original_shape[idx] for idx in range(1, self.ndim - 1)
            ] + [-1]
        else:
            starts = [0, 0] + dx[2:]
            ends = [-1, -1
                    ] + [original_shape[idx] for idx in range(2, self.ndim)]

        return tf.slice(output_tensor, starts, ends)
