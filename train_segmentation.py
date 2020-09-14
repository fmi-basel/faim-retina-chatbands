# coding: utf-8

import os
import tensorflow as tf
import keras
from absl import flags
from absl import app

from dlutils.layers.input_scaler import ScalingLayer
from dlutils.models.unet import GenericUnetBase

from retchat.callbacks import CosineAnnealingSchedule
from retchat.callbacks import ExtendedTensorBoard

from dlutils.video_utils.dataset import create_dataset_for_training
from dlutils.video_utils.dataset import create_dataset_for_validation
from dlutils.utils import configure_gpu_session
from dlutils.layers.padding import DynamicPaddingLayer
from dlutils.layers.padding import DynamicTrimmingLayer

from retchat.dataset import SegmentationRecordParser
from retchat.losses import get_masked_combined_mae_and_bce
from retchat.losses import get_masked_mae
from retchat.losses import get_masked_bce

from retchat.models import save_model_architecture
from retchat.layers import WeightedRegressionLayer
from retchat.layers import WeightedRegressionWithUncertaintyPredictionLayer
from retchat.augmentations import gaussian_noise, intensity_distortion

# Application parameters
ARGS = flags.FLAGS
flags.DEFINE_string(
    'dataset_folder', None, 'Folder with training dataset (as *tfrecords). '
    'Validation data is expected in the same format in the subfolder "validation".'
)
flags.mark_flag_as_required('dataset_folder')
flags.DEFINE_integer('batch_size', 32, 'Total batch size', lower_bound=1)
flags.DEFINE_integer('epochs', 10, 'Number of epochs to train', lower_bound=1)
flags.DEFINE_string('outdir', './models', 'output directory')
flags.DEFINE_integer('n_levels', 5, 'Number of levels')
flags.DEFINE_integer('downsampling', 1, 'Downsampling factor')
flags.DEFINE_float('width', 1.0, 'Width')
flags.DEFINE_float('lr_max', 1e-4, 'Learning rate maximum')
flags.DEFINE_float('lr_min', 1e-7, 'Learning rate minimum')
flags.DEFINE_float('dropout', 0, 'Dropout rate')
flags.DEFINE_string('model', 'unet', 'Backbone model name')
flags.DEFINE_boolean('verbose', False, 'Verbose mode')
flags.DEFINE_string('loss', None, 'loss function to use')
flags.DEFINE_integer('restarts', 1, 'Number of restarts')
flags.DEFINE_integer(
    'patience', 15,
    'Number of epochs without improvement before terminating early.')
flags.DEFINE_string('head', None, 'Regression head to be used.')
flags.DEFINE_float('alpha', None, 'Loss balance weight.')

# Configure tensorflow session. Necessary for multple users running on GPU.
configure_gpu_session()

INTENSITY_SCALING = {'lower': 0, 'upper': 30000}


def get_dataset(dataset_folder, batch_size):
    '''builds dataset iterators from tfrecords in the given folder.

    A subfolder containing the validation data is expected in
       <dataset_folder>/validation

    '''
    augmentations = [
        gaussian_noise(
            0,  # noise distributions' sigma mean
            10),  # noise distributions' sigma sigma
        intensity_distortion(
            0,
            20,  # offset mu/sigma
            1.0,
            0.05  # scale mu/sigma
        )
    ]

    parser = SegmentationRecordParser(fixed_ndim=3)

    dataset, training_steps = create_dataset_for_training(
        os.path.join(dataset_folder, '*.tfrecord'),
        batch_size=batch_size,
        parser_fn=parser.parse,
        augmentations=augmentations)

    validation_dataset, validation_steps = create_dataset_for_validation(
        os.path.join(dataset_folder, 'validation', '*.tfrecord'),
        batch_size=batch_size,
        parser_fn=parser.parse)

    with tf.device('/cpu:0'):
        iterator = dataset.make_one_shot_iterator()
        training_iter = iterator.get_next()

        iterator = validation_dataset.make_one_shot_iterator()
        validation_iter = iterator.get_next()

    return {
        'training_iter': training_iter,
        'validation_iter': validation_iter,
        'training_steps': training_steps,
        'validation_steps': validation_steps
    }


def construct_model(model_name, n_classes=2, n_channels=1, final_weights=None):
    '''
    '''

    def _construct():
        '''
        '''

        head_kernel = (3, 3)
        n_features = int(ARGS.width * 64 * n_classes * 2)
        downsampling_factor = ARGS.downsampling
        projection_axis = 2

        input_shape = (None, None, n_channels)
        input = keras.layers.Input(shape=input_shape)

        # Preprocessing
        # rescale to [0, 1] with clipping.
        scaled_input = ScalingLayer(**INTENSITY_SCALING)(input)
        if downsampling_factor > 1:
            input_shape = (None, None, n_features)
            scaled_input = DynamicPaddingLayer(factor=downsampling_factor,
                                               name='outer_pad')(scaled_input)
            scaled_input = keras.layers.Conv2D(
                n_features,
                kernel_size=downsampling_factor,
                activation='relu',
                strides=downsampling_factor)(scaled_input)
            # scaled_input = keras.layers.MaxPooling2D(downsampling_factor)(
            #     scaled_input)

        # create backbone model
        if model_name == 'unet':
            backbone = GenericUnetBase(
                input_shape=input_shape,
                n_levels=ARGS.n_levels,
                with_bn=True,
                width=ARGS.width,
                dropout=ARGS.dropout)
        else:
            raise RuntimeError('Unknown model backbone: {}'.format(model_name))

        backbone_output = backbone(scaled_input)

        intermediate = keras.layers.Conv2D(
            n_features, kernel_size=3, activation='relu',
            padding='same')(backbone_output)
        intermediate = keras.layers.BatchNormalization()(intermediate)
        intermediate = keras.layers.Conv2D(
            n_features, kernel_size=3, activation='relu',
            padding='same')(intermediate)
        prediction = keras.layers.Conv2D(
            1, kernel_size=3, activation='sigmoid',
            padding='same')(intermediate)

        # Upsample if needed.
        if downsampling_factor > 1:
            prediction = keras.layers.Conv2DTranspose(
                filters=1, kernel_size=5,
                strides=downsampling_factor, padding='same')(prediction)
            prediction = DynamicTrimmingLayer(name='outer_dtrim')([input, prediction])

        model = keras.models.Model([input], [prediction])
        return model

    model = _construct()

    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-5),
        loss='mae',
        metrics=[
            'mae',
        ])

    model.summary()
    if final_weights is not None:
        model.load_weights(final_weights)

    return model


def get_outdir(outdir, dataset_folder, model_name):
    '''
    '''
    base_outdir = os.path.abspath(outdir)
    dataset_folder = os.path.abspath(dataset_folder)
    dataset_name = os.path.basename(dataset_folder)
    if not os.path.basename(base_outdir) == dataset_name:
        base_outdir = os.path.join(base_outdir, dataset_name)

    def _propose():
        '''
        '''
        count = 0
        while True:
            yield os.path.join(
                base_outdir, 'segm', model_name + '-L{}-D{}-{}'.format(
                    ARGS.n_levels, ARGS.downsampling, ARGS.head),
                'run{:03}'.format(count))
            count += 1

    for outdir in _propose():
        try:
            os.makedirs(outdir)
            break
        except FileExistsError:
            pass

    print('Output is written to {}'.format(outdir))
    return outdir


def main(*args):
    '''
    '''

    dataset = get_dataset(
        dataset_folder=ARGS.dataset_folder, batch_size=ARGS.batch_size)

    model = construct_model(ARGS.model)

    outdir = get_outdir(ARGS.outdir, ARGS.dataset_folder, ARGS.model)
    save_model_architecture(model,
                            os.path.join(outdir, 'model_architecture.yaml'))

    callbacks = [
        keras.callbacks.LearningRateScheduler(
            CosineAnnealingSchedule(
                lr_max=ARGS.lr_max,
                lr_min=ARGS.lr_min,
                epoch_max=ARGS.epochs / ARGS.restarts,
                reset_decay=10.,
            )),
        keras.callbacks.ModelCheckpoint(
            os.path.join(outdir, 'model_best.h5'),
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
            period=1),
        keras.callbacks.ModelCheckpoint(
            os.path.join(outdir, 'model_latest.h5'),
            verbose=0,
            save_best_only=False,
            save_weights_only=True,
            period=1),
        ExtendedTensorBoard(
            log_dir=os.path.join(outdir, 'logs'),
            write_graph=True,
            write_grads=False,
            write_images=False),
    ]

    if ARGS.patience >= 1:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=0.001,
                patience=ARGS.patience,
                verbose=0,
                mode='auto',
                baseline=10.,
                restore_best_weights=False))

    model.fit(
        dataset['training_iter']['image'],
        dataset['training_iter']['targets'],
        steps_per_epoch=dataset['training_steps'],
        epochs=ARGS.epochs,
        verbose=1 if ARGS.verbose else 2,
        callbacks=callbacks,
        validation_data=(dataset['validation_iter']['image'],
                         dataset['validation_iter']['targets']),
        validation_steps=dataset['validation_steps'])


if __name__ == '__main__':
    app.run(main)
