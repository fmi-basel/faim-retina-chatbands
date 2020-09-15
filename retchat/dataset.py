import os
import re

from glob import glob

import numpy as np
import tensorflow as tf

from scipy.io import loadmat
from skimage.external.tifffile import imread

from dlutils.dataset.tfrec_utils import _int64_feature
from dlutils.dataset.tfrec_utils import _float_feature
from dlutils.dataset.tfrec_utils import _bytes_feature

UNLABELED = -1.


def label_to_image_fname(fname):
    '''
    '''

    result = re.sub('(.*)_w(.*)_\s+stack\s+(\d+)\s+.*', '\g<1>_w*_s\g<3>.stk',
                    fname)
    if result is None:
        raise ValueError('Could not substitute fname {}'.format(fname))
    return result


def _get_stack_id(path):
    '''
    '''
    fname = os.path.basename(path)
    match = re.search('stack\s+(\d+)\s+', fname)
    return int(match.group(1))


class DataFinder:
    '''
    '''

    def __init__(self,
                 data_dirs,
                 label_dir,
                 label_pattern='*.mat',
                 label_to_img_fn=None):
        '''
        '''
        self.data_dirs = data_dirs
        self.label_dir = label_dir
        self.label_pattern = label_pattern
        self.allowed_channels = ['confGFP', 'conf488']

    def image_from_label_path(self, label_path):
        '''
        '''
        fname = os.path.basename(label_path)
        image_fname = label_to_image_fname(fname)
        for data_dir in self.data_dirs:
            for image_path in glob(os.path.join(data_dir, image_fname)):
                if any(channel in os.path.basename(image_path)
                       for channel in self.allowed_channels):
                    return image_path
        return None

    def __iter__(self):
        '''
        '''
        for label_path in sorted(glob(
                os.path.join(self.label_dir, self.label_pattern)),
                                 key=_get_stack_id):
            image_path = self.image_from_label_path(label_path)
            if image_path is not None:
                yield label_path, image_path
            else:
                print('No file found for {} at {}'.format(
                    label_path, image_path))


class Stack:
    '''
    '''

    def __init__(self, path):
        '''
        '''
        self.data = np.expand_dims(np.moveaxis(imread(path), 0, -1), -1)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        '''
        '''
        return self.data.shape[0]

    @property
    def shape(self):
        '''
        '''
        return self.data.shape


class MaxBlockStack(Stack):
    '''
    '''

    def __init__(self, path, delta_x, **kwargs):
        '''
        '''
        super().__init__(path, **kwargs)
        self.dx = delta_x

    def __getitem__(self, idx):
        return self.data[max(0, idx -
                             self.dx):min(idx + self.dx +
                                          1, len(self.data))].max(axis=0)


class Max2R10PreprocessedStack(Stack):
    '''
    '''

    @staticmethod
    def _normalize(img, lower, upper):
        '''
        '''
        img = (img.astype(np.float32) - lower) / (upper - lower)
        img = (np.clip(img * 10000, 0,
                       np.iinfo(np.uint16).max)).astype(np.uint16)
        return img

    def _preprocess(self):
        '''
        '''
        from scipy.ndimage.filters import maximum_filter1d
        from scipy.ndimage.filters import rank_filter
        self.data = maximum_filter1d(self.data, axis=0, size=2 * self.dx + 1)

        kernel = np.ones((self.dx * 2 + 1, ) + (1, ) *
                         (len(self.data.shape) - 1))
        for _ in range(2):
            self.data = rank_filter(self.data, rank=10, footprint=kernel)

        lower, upper = np.percentile(self.data.flat, (5, 95))
        self.data = self._normalize(self.data, lower, upper)

    def __init__(self, path, delta_x, **kwargs):
        '''
        '''
        super().__init__(path, **kwargs)
        self.dx = delta_x
        self._preprocess()


class PercentileNormalizedMaxBlockStack(MaxBlockStack):
    '''
    '''

    @staticmethod
    def _normalize(img, lower, upper):
        '''
        '''
        img = (img.astype(np.float32) - lower) / (upper - lower)
        img = (np.clip(img * 10000, 0,
                       np.iinfo(np.uint16).max)).astype(np.uint16)
        return img

    def __init__(self, path, delta_x, percentiles, **kwargs):
        assert len(percentiles) == 2
        super().__init__(path=path, delta_x=delta_x, **kwargs)
        self.data = self._normalize(
            self.data, *np.percentile(self.data.flat, percentiles))


class RescalingNormalizedMaxBlockStack(MaxBlockStack):
    '''
    '''

    @staticmethod
    def _normalize(img, lower, upper):
        '''
        '''
        img = (img.astype(np.float32) - lower) / (upper - lower)
        img = (np.clip(img * 10000, 0,
                       np.iinfo(np.uint16).max)).astype(np.uint16)
        return img

    def __init__(self, path, delta_x, percentiles, z_factor, **kwargs):
        assert len(percentiles) == 2
        from skimage.transform import rescale

        super().__init__(path=path, delta_x=delta_x, **kwargs)
        self.data = rescale(self.data, (1, ) * 2 + (z_factor, ) + (1, ),
                            anti_aliasing=False)

        self.data = self._normalize(
            self.data, *np.percentile(self.data.flat, percentiles))


def load_annotation(label_path):
    '''
    '''
    data = loadmat(label_path)

    def _replace_nan(vals, new_value):
        '''in-place substitution of nan values.

        '''
        vals[np.isnan(vals)] = new_value
        return vals

    return {
        key: _replace_nan(data[key], UNLABELED)
        for key in ['chatpos1', 'chatpos2']
    }


class Annotation:
    def __init__(self, path):
        '''
        '''
        self.plane = load_annotation(path)
        for vals in self.plane.values():
            print(vals.shape, vals.min(), vals.max(), vals.dtype)

    def __getitem__(self, idx):
        '''
        '''
        if isinstance(idx, str):
            return self.plane[idx]

        return np.concatenate([
            self.plane[key][idx][..., None]
            for key in sorted(self.plane.keys())
        ],
                              axis=-1)

    @property
    def shape(self):
        '''
        '''
        return [self.plane[key].shape for key in sorted(self.plane.keys())]

    def keys(self):
        '''
        '''
        return sorted(self.plane.keys())

    def is_valid(self):
        '''
        '''
        return np.logical_and(*[val > 0 for val in self.plane.values()])


def annotation_to_segmentation(annotation, shape):
    '''
    '''
    position = np.ones(shape) * np.arange(0, shape[-1]).reshape((1, 1, -1))
    keys = list(annotation.keys())
    mask = np.logical_and(position >= annotation[keys[0]][..., None],
                          position <= annotation[keys[1]][..., None])
    return mask


class RegressionRecordParser:
    '''defines the serialize and parse function for an image
    regression dataset.

    '''

    def __init__(self, image_dtype=tf.uint16, fixed_ndim=None):
        '''
        '''
        self.image_dtype = image_dtype
        self.fixed_ndim = fixed_ndim
        assert self.fixed_ndim is None or self.fixed_ndim >= 2

    @staticmethod
    def serialize(frame, index, targets):
        '''serializes a training tuple such that it can be written as tfrecord example.

        '''

        features = tf.train.Features(
            feature={
                'img_shape': _int64_feature(list(frame.shape)),
                'target_shape': _int64_feature(list(targets.shape)),
                'index': _int64_feature(index),
                'image': _bytes_feature(frame.tostring()),
                'targets': _float_feature(list(targets.flatten()))
            })

        return tf.train.Example(features=features)

    def parse(self, example):
        '''parse a tfrecord example.

        '''
        features = {
            # Extract features using the keys set during creation
            'index': tf.FixedLenFeature([], tf.int64),
            'img_shape': tf.FixedLenSequenceFeature([], tf.int64, True),
            'target_shape': tf.FixedLenSequenceFeature([], tf.int64, True),
            'targets': tf.FixedLenSequenceFeature([], tf.float32, True),
            'image': tf.FixedLenFeature([], tf.string),
        }
        sample = tf.parse_single_example(example, features)

        # Fixed shape appears to be necessary for training with keras.
        if self.fixed_ndim is not None:
            shape = tf.reshape(sample['img_shape'], (self.fixed_ndim, ))
            target_shape = tf.reshape(sample['target_shape'],
                                      (self.fixed_ndim - 1, ))
        else:
            shape = sample['img_shape']
            target_shape = sample['target_shape']

        image = tf.decode_raw(sample['image'], self.image_dtype)
        image = tf.reshape(image, shape)
        image = tf.cast(image, tf.float32)
        targets = tf.reshape(sample['targets'], target_shape)
        return dict(targets=targets, index=sample['index'], image=image)


def _create_heatmap(target, shape):
    '''
    '''
    position = np.ones(shape) * np.arange(0, shape[-2]).reshape((1, -1, 1))
    heatmap = np.zeros(shape, dtype=np.float32)
    target = np.expand_dims(target, 1)
    target[target < 0] = 1000
    heatmap = np.expand_dims(
        np.exp(-0.1 * np.min(np.abs(target - position), axis=-1)), -1)
    return heatmap.astype(np.float32)


class SegmentationRecordParser(RegressionRecordParser):
    def parse(self, example):
        '''
        '''
        sample = super().parse(example)

        sample['targets'] = tf.py_func(
            _create_heatmap,
            [sample['targets'], tf.shape(sample['image'])], [tf.float32])
        sample['targets'] = tf.reshape(sample['targets'],
                                       tf.shape(sample['image']))

        return sample
