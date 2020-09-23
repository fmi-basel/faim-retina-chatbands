'''simple image data augmentation for training.
'''
import tensorflow as tf

IMG_KEY = 'image'
TARGET_KEY = 'targets'


def vertical_flip():
    '''flip up down with p=0.5

    '''
    return _cond_op(tf.image.flip_up_down, 0.5)


def horizontal_flip():
    '''flip left right with p=0.5

    '''
    return _cond_op(tf.image.flip_left_right, 0.5)


def _cond_op(transform, threshold):
    '''generates a conditional transform that is applied in
    a frequency proportional to threshold. threshold has to be within [0, 1]
    '''

    assert 0. <= threshold <= 1.0

    def _transform(input_dict):
        '''
        '''
        transform_prob = tf.random.uniform(shape=[],
                                           minval=0,
                                           maxval=1.,
                                           dtype=tf.float32)

        for key in [IMG_KEY, TARGET_KEY]:
            input_dict[key] = tf.cond(
                transform_prob >= threshold,  # condition
                lambda: transform(input_dict[key]),  # if
                lambda: input_dict[key])  # else

        return input_dict

    return _transform


def gaussian_noise(noise_mu, noise_sigma):
    '''generates a transform that adds iid gaussian noise to each pixel in
    the image.

    '''

    def _distorter(input_dict):
        '''
        '''
        image = input_dict[IMG_KEY]
        sigma = tf.maximum(
            0., tf.random_normal(shape=[], mean=noise_mu, stddev=noise_sigma))
        noise = tf.random_normal(shape=tf.shape(image), mean=0, stddev=sigma)
        input_dict[IMG_KEY] = image + noise
        return input_dict

    return _distorter


def intensity_distortion(offset_mu, offset_sigma, scale_mu, scale_sigma):
    '''generates a random intensity offset and scale distortion.

    '''

    def _distorter(input_dict):
        img = input_dict[IMG_KEY]
        offset = tf.random_normal(shape=[],
                                  mean=offset_mu,
                                  stddev=offset_sigma)
        scale = tf.random_normal(shape=[], mean=scale_mu, stddev=scale_sigma)
        input_dict[IMG_KEY] = img * scale + offset
        return input_dict

    return _distorter
