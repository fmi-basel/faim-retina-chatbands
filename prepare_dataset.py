# coding: utf-8

import os
from glob import glob
import warnings

from absl import flags
from absl import app
import numpy as np

from sklearn.model_selection import train_test_split

from retchat.dataset import DataFinder
from retchat.dataset import Annotation
from retchat.dataset import Stack, MaxBlockStack, PercentileNormalizedMaxBlockStack, Max2R10PreprocessedStack
from retchat.dataset import label_to_image_fname
from retchat.dataset import RegressionRecordParser
from retchat.sampler import PlaneSampler

from dlutils.video_utils.tfrec_utils import tfrecord_from_sampler

# Application parameters
ARGS = flags.FLAGS
flags.DEFINE_multi_string('data_folder', None, 'Folder containing stacks.')
flags.DEFINE_string('label_folder', None, 'Folder containing annotations.')
flags.DEFINE_string('output_folder', None, 'Folder to write tfrecords')
flags.DEFINE_integer('samples_per_stack',
                     500,
                     'Number of frames to sample from each stack.',
                     lower_bound=1)
flags.DEFINE_integer('validation',
                     2,
                     'Number of stacks to use for validation',
                     lower_bound=0)
flags.DEFINE_integer('delta',
                     5,
                     'Number of planes around central plane to project over.',
                     lower_bound=0)
flags.DEFINE_integer(
    'patch_size', 224,
    'Patch size. Will be applied in both x and y axis leading to square patches.'
)
flags.DEFINE_string('stack_suffix', '.stk', 'Suffix of videos')
flags.DEFINE_string('projector', None, 'Projector to use for preprocessing')

flags.DEFINE_string('labels_suffix', '.mat', 'Suffix of annotations')

for required_flag in ['data_folder', 'label_folder', 'output_folder']:
    flags.mark_flag_as_required(required_flag)


def _create_folder_if_needed(folder):
    '''
    '''
    if not os.path.exists(folder):
        print('Creating folders: {}'.format(folder))
        os.makedirs(folder)


def process(image_path, label_path, outdir):
    '''
    '''
    output = os.path.join(
        outdir,
        os.path.splitext(os.path.basename(image_path))[0] + '.tfrecord')

    parser = RegressionRecordParser()
    if ARGS.projector == 'percentile_max':
        stack = PercentileNormalizedMaxBlockStack(image_path,
                                                  delta_x=ARGS.delta,
                                                  percentiles=(5, 95))
    elif ARGS.projector == 'max':
        stack = MaxBlockStack(image_path, delta_x=ARGS.delta)
    elif ARGS.projector == 'max2rank':
        stack = Max2R10PreprocessedStack(image_path, delta_x=ARGS.delta)
    else:
        stack = Stack(image_path)

    sampler = PlaneSampler(stack=stack,
                           annotation=Annotation(label_path),
                           n_samples=ARGS.samples_per_stack,
                           patch_size=ARGS.patch_size)

    tfrecord_from_sampler(output, sampler, parser.serialize)


def split_dataset(vals, validation_samples, test_samples):
    '''splits videos into three categories: training, validation and test.

    '''
    splits = {}

    if validation_samples <= 0 and test_samples <= 0:
        splits['training'] = vals
        return splits

    splits['training'], remaining = train_test_split(
        sorted(vals), test_size=validation_samples + test_samples)
    if test_samples <= 0:
        splits['validation'] = remaining
        return splits
    splits['validation'], splits['test'] = train_test_split(
        remaining, test_size=test_samples)
    return splits


def main(*args):
    '''
    '''
    outdir = os.path.join(
        ARGS.output_folder,
        'N{}-{}'.format(ARGS.samples_per_stack,
                        ('single' if ARGS.projector is None else
                         '{}-block-{}'.format(ARGS.projector, ARGS.delta))))
    _create_folder_if_needed(outdir)
    print('Creating training data in {}'.format(outdir))

    # initialize seed
    np.random.seed(13)

    # prepare samples from individual videos.

    paths = list(
        DataFinder(data_dirs=ARGS.data_folder,
                   label_dir=ARGS.label_folder,
                   label_pattern='*' + ARGS.labels_suffix,
                   label_to_img_fn=label_to_image_fname))
    if not paths:
        print(
            'No matching files found for data_folder: {} and label_folder: {}'.
            format(ARGS.data_folder, ARGS.label_folder))
        return

    splits = split_dataset(paths, ARGS.validation, 0)

    for split in splits:
        print(split, splits[split])

    for split in splits.keys():
        if split == 'training':
            split_outdir = outdir
        else:
            split_outdir = os.path.join(outdir, split)
        _create_folder_if_needed(split_outdir)

        for label_path, image_path in splits[split]:
            process(image_path, label_path, split_outdir)


if __name__ == '__main__':
    app.run(main)
