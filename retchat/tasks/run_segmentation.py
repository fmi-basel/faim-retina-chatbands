import logging
import os

import luigi
from luigi.util import requires

from skimage.external import tifffile

import pandas
import numpy as np

from scipy.ndimage.filters import maximum_filter1d
from skimage.transform import downscale_local_mean
from skimage.transform import resize

from retchat.models import load_model
from keras.backend import clear_session

from .collector import parse_meta
from .collector import imread_raw
from .collector import ChatbandStackCollectorTask

logger = logging.getLogger('luigi-interface')

TARGET_SPACING = (2.07e-7, 2.07e-7, 3e-7, 1)


def predict_complete(model, stack):
    '''run model plane-wise on given stack. Output is a probability stack
    scaled to [0, 255] encoded as uint8.
    '''
    probs = np.asarray([
        model.predict(np.expand_dims(stack[idx], 0))
        for idx in range(len(stack))
    ])
    probs = (np.clip(probs, 0, 1.) * 255).astype(np.uint8)
    probs = probs.squeeze(axis=1)
    return probs


def save_segmentation(output_target, probs):
    '''save probs as tiff stack.
    '''
    with output_target.temporary_path() as fout:
        logger.debug('Saving to %s', output_target.fn)
        tifffile.imsave(fout, probs, compress=6)


def _normalize(img, lower, upper):
    '''normalize image from [lower, upper] to [0, 10000].
    '''
    delta = max(upper - lower, 1.)  # avoid div by 0.
    img = (img.astype(np.float32) - lower) / delta
    img = (np.clip(img * 10000, 0, np.iinfo(np.uint16).max)).astype(np.uint16)
    return img


def _project(img, dx, axis=0):
    '''rolling max projection within a window of 2*dx + 1 along axis.
    '''
    return maximum_filter1d(img, axis=axis, size=2 * dx + 1)


class Preprocessor:
    '''
    '''
    def __init__(self, target_spacing, percentiles, dx=10):
        '''
        '''
        self.target_spacing = np.asarray(target_spacing)
        self.percentiles = percentiles
        self.dx = dx
        assert all(0 <= val <= 100 for val in self.percentiles)

    def run(self, stack):
        '''
        '''
        logger.debug('Resampling stack...')
        is_resampled = self._resample(stack)
        logger.debug('Resampling {}'.format('done. Result has shape={}'.format(
            stack.data.shape) if is_resampled else 'not required'))

        logger.debug('Normalizing to {} percentiles...'.format(
            self.percentiles))
        stack.data = _normalize(
            stack.data, *np.percentile(stack.data.flat, self.percentiles))
        logger.debug('Normalization done.')

        # and do the max-projection
        logger.debug('Applying max-projection...')
        stack.data = _project(stack.data, dx=self.dx)
        logger.debug('Max projection done.')

    def _resample(self, stack):
        '''
        '''
        factors = self.target_spacing / np.asarray(stack.spacing)

        logger.debug('  target.spacing={}'.format(self.target_spacing))
        logger.debug('  stack.spacing= {}'.format(stack.spacing))
        logger.debug('  factor=        {}'.format(factors))

        if np.all(np.abs(factors - 1.) / factors < 0.25):
            logger.debug('  resampling not required...')
            return False

        if np.all((np.round(factors, 0) - factors) /
                  factors < 0.2) and not np.any(factors < 0.8):
            logger.debug('  using skimage.transform.downscale_local_mean...')
            stack.data = downscale_local_mean(
                stack.data, tuple(int(val) for val in np.round(factors)))
        # else:
        #     # NOTE 'rescale' expects the factor to calculate the
        #     # number of resulting pixels, we need to give it the
        #     # *inverse* of our factors!
        #     logger.debug('  using skimage.transform.rescale...')
        #     stack.data = rescale(stack.data, 1. / factors, preserve_range=True, multichannel=False)
        return True

    @staticmethod
    def resample_back(stack, prediction):
        '''
        '''
        if np.all(stack.original_shape == prediction.shape):
            logger.debug('Resampling prediction not required...')
            return prediction
        logger.debug('Resampling prediction from {} to {}'.format(
            prediction.shape, stack.original_shape))
        return resize(prediction, stack.original_shape, preserve_range=True)


class Stack:
    def __init__(self, path):
        '''
        '''
        # load and move Z to last dimension.
        self.data = imread_raw(path)
        logger.info('loaded image: {}'.format(self.data.shape))
        self.data = np.expand_dims(np.moveaxis(self.data, 0, -1), -1)

        self.meta = parse_meta(path)
        self.original_shape = self.data.shape

    @property
    def spacing(self):
        '''
        '''
        return tuple(self.meta[key]
                     for key in ['scale_x', 'scale_y', 'scale_z']) + (1, )

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]


@requires(ChatbandStackCollectorTask)
class ChatbandSegmentationTask(luigi.Task):
    '''applies the trained segmentation model on a series of inputs.
    '''
    task_namespace = 'retina'

    output_folder = luigi.Parameter()
    '''output folder into which the segmentations are written.
    '''

    split_idx = luigi.IntParameter(default=0)
    '''determines which split is processed.
    '''
    split_fraction = luigi.IntParameter(default=1)
    '''number of splits of the inputs to process.

    E.g. split_fraction=5 with 100 inputs would generate 5 jobs each
    processing 20 stacks.
    '''

    model_dir = luigi.Parameter()
    '''folder containing model to use.
    '''

    model_weights_fname = luigi.Parameter(default='model_latest.h5')
    '''file name of weights to use.
    '''

    model_architecture_fname = luigi.Parameter(
        default='model_architecture.yaml')
    '''file name of architecture to use.
    '''
    @property
    def input_files(self):
        '''returns paths for all stacks in the given split.
        '''
        with self.input().open('r') as fin:
            df = pandas.read_csv(fin)
        paths = df['path'].values.tolist()
        offset = int(np.ceil(len(paths) / self.split_fraction))
        start = self.split_idx * offset
        end = min((self.split_idx + 1) * offset, len(paths))
        logger.debug('Processing stacks of indices {} to {}'.format(
            start, end))
        return paths[start:end]

    def run(self):
        '''
        '''
        model = load_model(
            os.path.join(self.model_dir, self.model_architecture_fname),
            os.path.join(self.model_dir, self.model_weights_fname))

        preprocessor = Preprocessor(target_spacing=TARGET_SPACING,
                                    percentiles=(5, 95),
                                    dx=10)

        errors = []

        for input_file, output_file in zip(self.input_files, self.output()):

            if output_file.exists():
                logger.debug('Target %s already exists.', output_file.fn)
                continue

            try:
                stack = Stack(input_file)
                preprocessor.run(stack)

                logger.debug('Running prediction...')
                probs = predict_complete(model, stack)
                logger.debug('Prediction done.')

                save_segmentation(output_file, probs)

            except Exception as err:
                logger.error('Error encountered for image %s: %s',
                             input_file,
                             err,
                             exc_info=True)
                errors.append(err)

        if len(errors):
            raise RuntimeError('Encountered {} errors:'.format(len(errors)) +
                               '\n'.join(str(err) for err in errors))
        clear_session()

    def complete(self):
        '''
        '''
        if not self.input().exists():
            return False

        return super().complete()

    def output(self):
        '''
        '''
        def _get_output(input_file):
            _, fname = os.path.split(input_file)
            fname = os.path.splitext(fname)[0] + '.tif'
            return luigi.LocalTarget(os.path.join(self.output_folder, fname))

        return [_get_output(input_file) for input_file in self.input_files]


class ParallelChatbandPredictionTask(luigi.WrapperTask):
    '''Distributes segmentation of many stacks over multiple jobs.
    '''

    task_namespace = 'retina'

    output_fname = luigi.Parameter(default='stacks_to_process.txt')
    '''file name of list containing all stacks to process. This will
    be generated by the collector task.
    '''

    input_folder = luigi.Parameter()
    '''input folder which is scanned to identify stacks to process.
    '''

    output_folder = luigi.Parameter()
    '''output folder into which the segmentations are written.
    '''

    split_idx = luigi.IntParameter(default=0)
    '''determines which split is processed.
    '''
    split_fraction = luigi.IntParameter(default=1)
    '''number of splits of the inputs to process.

    E.g. split_fraction=5 with 100 inputs would generate 5 jobs each
    processing 20 stacks.

    You should not choose split_fraction larger than the number of
    files to process.

    '''

    model_dir = luigi.Parameter()
    '''folder containing model to use.
    '''

    model_weights_fname = luigi.Parameter(default='model_latest.h5')
    '''file name of weights to use.
    '''

    model_architecture_fname = luigi.Parameter(
        default='model_architecture.yaml')
    '''file name of architecture to use.
    '''
    def requires(self):
        '''
        '''
        return [
            ChatbandSegmentationTask(
                split_idx=idx,
                split_fraction=self.split_fraction,
                output_folder=self.output_folder,
                input_folder=self.input_folder,
                output_fname=self.output_fname,
                model_dir=self.model_dir,
                model_weights_fname=self.model_weights_fname,
                model_architecture_fname=self.model_architecture_fname)
            for idx in range(self.split_fraction)
        ]


@requires(ChatbandStackCollectorTask)
class VisualiseSegmentationTask(luigi.Task):
    '''
    '''

    task_namespace = 'retina'

    number_of_stacks = luigi.IntParameter(default=10)
    number_of_slices = luigi.IntParameter(default=10)

    def run(self):
        '''
        '''
        logger = logging.getLogger('luigi-interface')

        def _get_output(input_file):
            _, fname = os.path.split(input_file)
            fname = os.path.splitext(fname)[0] + '.tif'
            return luigi.LocalTarget(os.path.join(self.output_folder, fname))

        with self.input().open('r') as fin:
            df = pandas.read_csv(fin)

        np.random.seed(13)

        paths = df['path'].sample(n=self.number_of_stacks, replace=False)
        paths = paths.values.tolist()

        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        errors = []

        with self.output().temporary_path() as output_path:
            with PdfPages(output_path) as pdf:
                for path in paths:

                    try:
                        logger.debug('{}'.format(os.path.basename(path)))

                        img = imread_raw(path)
                        segm = imread_raw(_get_output(path).fn)

                        logger.debug('  img.shape={}, segm.shape={}'.format(
                            img.shape, segm.shape))

                        for idx in sorted(
                                np.random.choice(len(segm),
                                                 size=self.number_of_slices,
                                                 replace=False)):

                            img_plane = img[:, idx, ...].squeeze()
                            vmin, vmax = np.percentile(img_plane.flat, (5, 95))
                            aspect = img_plane.shape[1] / img_plane.shape[-1]

                            segm_plane = segm[idx, ...].squeeze().T

                            axarr = plt.subplots(2, 1, figsize=(16, 6))[1]
                            axarr[0].imshow(img_plane,
                                            cmap='Greys',
                                            vmin=vmin,
                                            vmax=vmax,
                                            aspect=aspect)
                            axarr[1].imshow(segm_plane,
                                            vmin=0,
                                            vmax=255,
                                            aspect=aspect)
                            axarr[0].set_title('{}: \nplane {}'.format(
                                os.path.basename(path), idx))
                            plt.tight_layout()
                            pdf.savefig(bbox_inches='tight')
                            plt.close()
                    except Exception as err:
                        errors.append(err)
                        logger.error('Failed for {}: {}'.format(path, err))

        if errors:
            raise RuntimeError('Encountered {} errors:\n\t' +
                               '\n\t'.join(str(err) for err in errors))

    def output(self):
        '''
        '''
        return luigi.LocalTarget(
            os.path.join(
                self.output_folder, 'vis',
                'segmentation_samples_n{}_s{}.pdf'.format(
                    self.number_of_stacks, self.number_of_slices)))
