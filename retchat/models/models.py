import functools
import re

import keras_applications
import keras

keras_applications.set_keras_submodules(backend=keras.backend,
                                        layers=keras.layers,
                                        utils=keras.utils,
                                        models=keras.models)

ResNet50 = keras.applications.ResNet50
ResNet101 = keras_applications.resnet.ResNet101
ResNet152 = keras_applications.resnet.ResNet152

ResNet50V2 = keras_applications.resnet_v2.ResNet50V2
ResNet101V2 = keras_applications.resnet_v2.ResNet101V2
ResNet152V2 = keras_applications.resnet_v2.ResNet152V2
MobileNetV2 = keras_applications.mobilenet_v2.MobileNetV2

MODEL_NAMES = {
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'resnet152': ResNet152,
    'resnet50v2': ResNet50V2,
    'resnet101v2': ResNet101V2,
    'resnet152v2': ResNet152V2,
    'mobilenetv2': functools.partial(MobileNetV2, alpha=1.4),
}


def VariableWidthResNet50(width=1.0,
                          include_top=True,
                          weights='imagenet',
                          input_tensor=None,
                          input_shape=None,
                          pooling=None,
                          classes=1000,
                          **kwargs):
    '''variable width resnet50.

    '''
    from keras_applications.resnet_common import stack1
    from keras_applications.resnet_common import ResNet

    def stack_fn(x):
        x = stack1(x, int(width * 64), 3, stride1=1, name='conv2')
        x = stack1(x, int(width * 128), 4, name='conv3')
        x = stack1(x, int(width * 256), 6, name='conv4')
        x = stack1(x, int(width * 512), 3, name='conv5')
        return x

    return ResNet(stack_fn, False, True, 'resnet50w{:1.02f}'.format(width),
                  include_top, weights, input_tensor, input_shape, pooling,
                  classes, **kwargs)


def VariableWidthResNet50V2(width=1.0,
                            include_top=True,
                            weights='imagenet',
                            input_tensor=None,
                            input_shape=None,
                            pooling=None,
                            classes=1000,
                            **kwargs):
    '''variable width resnet50v2.

    '''
    from keras_applications.resnet_common import stack2
    from keras_applications.resnet_common import ResNet

    def stack_fn(x):
        x = stack2(x, int(width * 64), 3, name='conv2')
        x = stack2(x, int(width * 128), 4, name='conv3')
        x = stack2(x, int(width * 256), 6, name='conv4')
        x = stack2(x, int(width * 512), 3, stride1=1, name='conv5')
        return x

    return ResNet(stack_fn, True, True, 'resnet50v2w{:1.02f}'.format(width),
                  include_top, weights, input_tensor, input_shape, pooling,
                  classes, **kwargs)


VARIABLE_MODEL_NAMES = {
    'resnet50': VariableWidthResNet50,
    'resnet50v2': VariableWidthResNet50V2,
}


def parse_from_name(name):
    '''
    '''
    match = re.search('(.*)[Ww](\d+\.\d+)$', name)
    model_name = match.group(1)
    width = float(match.group(2))
    return model_name, width


def get_model_by_name(model_name):
    '''returns the model constructor.

    '''
    try:
        return MODEL_NAMES[model_name.lower()]
    except KeyError:
        pass

    try:
        submodel_name, width = parse_from_name(model_name)
        return functools.partial(VARIABLE_MODEL_NAMES[submodel_name],
                                 width=width)
    except (AttributeError, KeyError, ValueError):
        pass

    raise NotImplementedError('Model {} is not known. Choose from: {}'.format(
        model_name, ','.join(MODEL_NAMES.keys())))
