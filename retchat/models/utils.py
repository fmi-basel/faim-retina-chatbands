import os
import yaml

from dlutils.layers.input_scaler import ScalingLayer
from dlutils.layers.padding import DynamicPaddingLayer, DynamicTrimmingLayer

CUSTOM_LAYERS = {
    cls.__name__: cls
    for cls in [
        ScalingLayer, DynamicPaddingLayer,
        DynamicTrimmingLayer
    ]
}


def get_number_of_output_channels(model):
    '''returns the number of output channels.

    NOTE If more than one outputs are present, it returns the number
    of channels of the first.

    '''
    if len(model.outputs) == 1:
        return model.output_shape[-1]
    return model.output_shape[0][-1]


def get_number_of_input_channels(model):
    '''returns the number of input channels.

    NOTE If more than one inputs are present, it returns the number
    of channels of the first.

    '''
    if len(model.inputs) == 1:
        return model.input_shape[-1]
    return model.input_shape[0][-1]


def save_model_architecture(model, path):
    '''saves model architecutre as yaml.

    '''
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(path, 'w') as fout:
        fout.write(model.to_yaml())


def model_from_yaml(yaml_string):
    '''re-builds a model from it's configuration.

    '''
    config = yaml.full_load(yaml_string)
    from keras.layers import deserialize
    return deserialize(config, custom_objects=CUSTOM_LAYERS)


def load_model(architecture_path, weights_path):
    '''re-builds model and loads weights.

    '''
    with open(architecture_path, 'r') as fin:
        model = model_from_yaml(fin.read())
    model.load_weights(weights_path)
    return model
