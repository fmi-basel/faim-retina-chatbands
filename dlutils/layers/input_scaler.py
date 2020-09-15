from keras import layers
import tensorflow as tf


class ScalingLayer(layers.Layer):
    '''rescales the input tensors values from [lower, upper] to [0, 1.].
    Values outside are clipped.

    '''

    def __init__(self, lower, upper, **kwargs):
        '''
        '''
        super().__init__(**kwargs)
        assert upper > lower
        self.lower = lower
        self.delta = upper - lower

    def call(self, x):
        '''
        '''
        return tf.clip_by_value((x - self.lower) / self.delta, 0., 1.)

    def compute_output_shape(self, input_shape):
        '''
        '''
        return input_shape

    def get_config(self):
        '''
        '''
        config = super().get_config()
        config['lower'] = self.lower
        config['upper'] = self.lower + self.delta
        return config
