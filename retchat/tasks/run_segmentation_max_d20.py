'''tasks for running a trained model on new data.

'''
import logging
import os

import luigi
from luigi.util import requires
import pandas
import numpy as np

from retchat.models import load_model
from keras.backend import clear_session

from .ioutils import imread_raw
from .predutils import Preprocessor
from .predutils import predict_complete
from .predutils import save_segmentation
from .predutils import Stack

from .collector import ChatbandStackCollectorTask
from .collector import DEFAULT_FNAME_PATTERNS

logger = logging.getLogger('luigi-interface')

# hard-wired parameters for preprocessor. If the
# ChatbandSegmentationTask is to be used with different preprocessors,
# then these should be moved to its parameters or into an extra target
# from upstream.
TARGET_SPACING = (2.07e-7, 2.07e-7, 3e-7, 1)
NORMALIZATION_PERCENTILES = (0, 100)
ROLLING_MAX_DX = 20


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

        def _get_nth_chunk(nth, total_items, num_chunks):
            offset = total_items / num_chunks
            indices = np.round(np.linspace(0, total_items,
                                           num_chunks + 1)).astype(int)
            return indices[[nth, nth + 1]]

        start, end = _get_nth_chunk(self.split_idx, len(paths),
                                    self.split_fraction)

        if start == end:
            raise RuntimeError(
                'Attempted to process empty chunk. Decrease the split_fraction to process in larger chunks'
            )

        logger.debug('Processing stacks of indices {} to {}'.format(
            start, end))
        return paths[start:end]

    def run(self):
        '''
        '''
        model = load_model(
            os.path.join(self.model_dir, self.model_architecture_fname),
            os.path.join(self.model_dir, self.model_weights_fname))

        # NOTE preprocessor parameters are currently hard-wired for
        # models trained with percentile_max-block-10 preprocessing.
        preprocessor = Preprocessor(target_spacing=TARGET_SPACING,
                                    percentiles=NORMALIZATION_PERCENTILES,
                                    dx=ROLLING_MAX_DX)

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

    # These are the parameters inherited from the upstream collector
    # task.
    input_folder = luigi.Parameter()
    '''input folder which is scanned to identify stacks to process.
    '''
    fname_patterns = luigi.ListParameter(default=DEFAULT_FNAME_PATTERNS)
    '''list of file patterns matching stacks that need to be processed.
    E.g. *conf488*stk'''
    output_folder = luigi.Parameter()
    '''output folder into which the segmentations are written.
    '''
    output_fname = luigi.Parameter(default='stacks_to_process.txt')
    '''file name of list containing all stacks to process. This will
    be generated by the collector task.
    '''

    # These are the parameters specific for ChatbandSegmentationTask
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

    # ...and these are the parameters specific for
    # parallelization/chunking.
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
    '''generates a PDF with randomly sampled examples of input planes
    and segmentations. This works only if ChatbandSegmentationTask or
    ParallelChatbandPredictionTask has been run already.

    NOTE if used frequently, consider implementing the dependency on
    ParallelChatbandPredictionTask explicitly.

    '''

    task_namespace = 'retina'

    number_of_stacks = luigi.IntParameter(default=10)
    '''number of stacks to sample.
    '''
    number_of_slices = luigi.IntParameter(default=10)
    '''number of slices per stack to sample.
    '''

    def run(self):
        '''
        '''
        logger = logging.getLogger('luigi-interface')

        def _get_output(input_file):
            '''guesses the location of the segmentation generated by
            ChatbandSegmentationTask and provides the target for
            future loading.

            '''
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
