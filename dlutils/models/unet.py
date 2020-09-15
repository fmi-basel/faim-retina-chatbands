'''simple implementation of unet in keras. For a more generic and
flexible version, see more recent versions of dlutils.
'''
from keras.engine.topology import get_source_inputs

from keras.engine import Input
from keras.engine import Model
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Deconvolution2D
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import concatenate

from dlutils.layers.padding import DynamicPaddingLayer, DynamicTrimmingLayer


def get_model_name(width, n_levels, dropout, with_bn, *args, **kwargs):
    '''
    '''
    name = 'UNet-{}-{}'.format(width, n_levels)
    if with_bn:
        name += '-BN'
    if dropout is not None:
        name += '-D{}'.format(dropout)
    return name


def unet_block(base_features, n_levels, n_blocks_per_level, dropout,
               base_block, **block_kwargs):
    '''
    '''

    block_params = dict(padding='same', activation='relu')
    block_params.update(block_kwargs)

    pooling = MaxPooling2D
    upsampling = Deconvolution2D

    def features_from_level(level):
        features_out = base_features * 2**level
        return features_out

    def _get_name(prefix, level, suffix):
        return '{prefix}_L{level:02}_{suffix}'.format(
            prefix=prefix, level=level, suffix=suffix)

    def block(input_tensor):
        '''
        '''
        links = []

        x = input_tensor

        # contracting path.
        prefix = 'CB'
        for level in range(n_levels - 1):
            for count in range(n_blocks_per_level):
                x = base_block(
                    filters=features_from_level(level),
                    name=_get_name(prefix, level, 'C%d' % count),
                    **block_params)(x)

            if dropout > 0.:
                x = Dropout(dropout, name=_get_name(prefix, level, 'DROP'))(x)

            links.append(x)
            x = pooling(2, name=_get_name(prefix, level, 'MP'))(x)

        # compressed representation
        for count in range(n_blocks_per_level):
            x = base_block(
                filters=features_from_level(n_levels - 1),
                name=_get_name(prefix, n_levels - 1, 'C%d' % count),
                **block_params)(x)

        if dropout > 0.:
            x = Dropout(
                dropout, name=_get_name(prefix, n_levels - 1, 'DROP'))(x)

        # expanding path.
        prefix = 'EB'
        for level in reversed(range(n_levels - 1)):
            x = upsampling(
                filters=features_from_level(level),
                strides=2,
                kernel_size=2,
                name=_get_name(prefix, level, 'DC'))(x)
            x = concatenate(
                [x, links[level]], name=_get_name(prefix, level, 'CONC'))

            for count in range(n_blocks_per_level):
                x = base_block(
                    filters=features_from_level(level),
                    name=_get_name(prefix, level, 'C%d' % (count + 1)),
                    **block_params)(x)

            if dropout > 0.:
                x = Dropout(dropout, name=_get_name(prefix, level, 'DROP'))(x)
        return x

    return block


def ConvolutionWithBatchNorm(name,
                             conv=Convolution2D,
                             activation='relu',
                             **conv_kwargs):
    '''
    '''

    def block(input_tensor):
        x = conv(**conv_kwargs, name=name + '_cnv')(input_tensor)
        x = BatchNormalization(name=name + '_bn')(x)
        return x

    return block


def GenericUnetBase(input_shape=None,
                    input_tensor=None,
                    batch_size=None,
                    dropout=None,
                    with_bn=False,
                    width=1,
                    n_levels=5,
                    n_blocks=2):
    '''Constructs a basic U-Net.

    Based on: Ronneberger et al. "U-Net: Convolutional Networks for
    BiomedicalImage Segmentation", MICCAI 2015

    '''
    base_features = int(width * 64)

    # Assemble input
    # NOTE we use flexible sized inputs per default.
    if input_tensor is None:
        img_input = Input(
            batch_shape=(batch_size, ) + (None, None, input_shape[-1]),
            name='input')
    else:
        img_input = input_tensor

    x = DynamicPaddingLayer(factor=2**n_levels, name='dpad')(img_input)

    if with_bn:
        base_block = ConvolutionWithBatchNorm
    else:
        base_block = Convolution2D

    x = unet_block(
        dropout=dropout,
        n_levels=n_levels,
        base_features=base_features,
        n_blocks_per_level=n_blocks,
        base_block=base_block,
        kernel_size=3)(x)

    x = DynamicTrimmingLayer(name='dtrim')([img_input, x])

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    return Model(
        inputs=inputs,
        outputs=x,
        name=get_model_name(
            width=width,
            n_levels=n_levels,
            n_blocks=n_blocks,
            dropout=dropout,
            with_bn=False))


# for backwards compatibility
UnetBase = GenericUnetBase
