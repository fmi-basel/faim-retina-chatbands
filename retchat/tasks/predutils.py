import logging
import numpy as np

from skimage.external import tifffile

from scipy.ndimage.filters import maximum_filter1d
from skimage.transform import downscale_local_mean
from skimage.transform import resize

from .ioutils import parse_meta
from .ioutils import imread_raw

logger = logging.getLogger('luigi-interface')


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
    def __init__(self, target_spacing, percentiles, dx): # was dx=10
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

        if np.all(np.abs(factors - 1.) /
                  factors < 0.25):  # we give it some slack before
                                    # resampling is considered
                                    # necessary
            logger.debug('  resampling not required...')
            return False

        if np.all((np.round(factors, 0) - factors) /
                  factors < 0.2) and not np.any(factors < 0.8):
            logger.debug('  using skimage.transform.downscale_local_mean...')
            stack.data = downscale_local_mean(
                stack.data, tuple(int(val) for val in np.round(factors)))
        else:
            logger.debug('Not downscaled...')
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
