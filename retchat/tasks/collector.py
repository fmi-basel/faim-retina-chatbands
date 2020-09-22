import os
import glob
import itertools
import logging
import re
from xml.etree import ElementTree

import czifile
from skimage.external import tifffile

import luigi
import numpy as np
import pandas


logger = logging.getLogger('luigi-interface')


# Typical file extensions of stacks that need to be processed.
DEFAULT_FNAME_PATTERNS = [
        '*confGFP*stk',
        '*conf488*stk',
        '*.czi']


def parse_meta(path):
    '''convenience function to get meta data.
    '''
    ext = os.path.splitext(path)[1]
    if ext.lower() == '.czi':
        return CziParser.parse(path)
    elif ext.lower() == '.stk':
        return StkParser.parse(path)
    raise RuntimeError('Unknown file format')


def imread_raw(path):
    '''convenience function to read image stack.
    '''
    ext = os.path.splitext(path)[1]
    if ext.lower() == '.stk' or ext.lower() in ['.tif', '.tiff']:
        return tifffile.imread(path)
    elif ext.lower() == '.czi':
        img = czifile.imread(path).squeeze()

        meta = CziParser.parse_channel_meta(path)

        channel = None
        for key, val in meta.items():
            if val == 488:
                channel = int(re.search('T(\d)', key).group(1)) - 1

        if channel is None:
            logger.debug(meta)
            channel = 1
            logger.debug('Guessing green channel to be {}'.format(channel))

        logger.debug('Found green channel in {}'.format(channel))
        img = img[channel]
        logger.warn('Loading only channel 1 from czi!')
        return img
    raise NotImplementedError('Unknown extension: {}'.format(ext))


class CziParser:
    '''
    '''

    @staticmethod
    def parse(path):
        '''
        '''
        with czifile.CziFile(path) as fin:
            info = {
                'path': path,
                'shape': fin.shape,
                'axes': fin.axes,
                'dtype': fin.dtype
            }
            meta = fin.metadata()
            info.update(CziParser._parse_scales(meta))
            #info.update(CziParser._parse_channel_meta(meta))
        return info

    @staticmethod
    def _parse_scales(metadata):
        '''
        '''
        matches = re.findall('<Scaling([XYZ])>(.*)</Scaling([XYZ])>', metadata)
        return {
            'scale_' + axis.lower(): float(scale)
            for axis, scale, _ in matches
        }

    @staticmethod
    def _parse_channel_meta(metadata):
        '''
        '''
        info = {}
        xml_root = ElementTree.fromstring(metadata)
        for channel in xml_root.iter('Channel'):
            wavelengths = list(channel.findall('ExcitationWavelength'))
            if len(wavelengths) >= 2:
                raise RuntimeError('Multiple wavelenghts found for channel {}!'.format(
                    channel.get('Name')))
            if not wavelengths:
                continue
            info['channel_' + channel.get('Name')] = round(float(wavelengths[0].text))
        return info

    @staticmethod
    def parse_channel_meta(path):
        with czifile.CziFile(path) as fin:
            return CziParser._parse_channel_meta(fin.metadata())


class StkParser:
    @staticmethod
    def parse(path):
        '''
        '''
        with tifffile.TiffFile(path) as fin:
            info = {
                'path': path,
                'shape': fin.series[0].shape,
                'axes': fin.series[0].axes,
                'dtype': fin.series[0].dtype,
            }
            info.update(**StkParser._parse_scales(fin, path))
        return info

    @staticmethod
    def _parse_scales(filehandle, path):
        '''this is a quick and dirty hack for now :(

        '''
        scales = {}
        try:
            scales['scale_z'] = StkParser._find_zstep(
                StkParser._find_ndfile(path))
        except RuntimeError as err:
            logging.getLogger('luigi-interface').error(
                '{}. Using 0.3 um as z-step size.'.format(err))

        shape = filehandle.series[0].shape[-2:]
        if np.all(shape == (2048, 2048)):
            in_plane = 0.103 * 1e-6
        elif np.all(shape == (1200, 1200)):
            in_plane = 0.168 * 1e-6
        else:
            raise RuntimeError('Unknown XY setting for .stk: {}'.format(path))

        scales['scale_x'] = in_plane
        scales['scale_y'] = in_plane

        return scales

    @staticmethod
    def _find_zstep(ndfilepath):
        with open(ndfilepath, 'r') as fin:
            print(ndfilepath)
            txt = fin.read()
            return float(re.search('"ZStepSize", (\d+.\d+)', txt).group(1)) * 1e-6

    @staticmethod
    def _find_ndfile(stkpath):
        '''tries to identify the .nd file for a given .stk
        '''
        dirname, fname = os.path.split(stkpath)
        candidates = glob.glob(os.path.join(dirname, '*nd'))

        if len(candidates) <= 0:
            raise RuntimeError(
                'Could not find any .nd file for {}!'.format(stkpath))

        candidates = sorted(
            candidates,
            key=
            lambda candidate: len(os.path.commonprefix([stkpath, candidate])))

        if len(os.path.commonprefix([stkpath, candidates[-1]
                                     ])) <= len(dirname):
            raise RuntimeError(
                'Could not find matching .nd file for {}!'.format(stkpath))

        return candidates[-1]


class ChatbandStackCollectorTask(luigi.Task):
    '''This task collects all images to be processed and writes them to
    a .txt. Subsequent tasks (like the segmentation) will then use this
    list to schedule the work.

    '''
    output_folder = luigi.Parameter()
    '''output directory. Note that this parameter is typically shared
    with downstream tasks.
    '''
    output_fname = luigi.Parameter(default='stacks_to_process.txt')
    '''filename of output. This is a list of paths to files that need
    to be processed.
    '''
    input_folder = luigi.Parameter()
    '''folder to be scanned for files that need to be processed.
    '''
    fname_patterns = luigi.ListParameter(default=DEFAULT_FNAME_PATTERNS)
    '''list of file patterns matching stacks that need to be processed.
    E.g. *conf488*stk

    '''

    logger = logging.getLogger('luigi-interface')

    def run(self):
        '''
        '''
        self.logger.info('Start collecting files...')
        df = pandas.DataFrame([
            parse_meta(path) for path in itertools.chain.from_iterable((
                glob.glob(
                    os.path.join(self.input_folder, pattern), recursive=True)
                for pattern in self.fname_patterns))
        ])

        self.logger.info(
            '\n***\nFound {} files matching the given patterns.\n***'.format(
                len(df)))

        if len(df) <= 0:
            raise RuntimeError('Could not find any matching file!')

        with self.output().temporary_path() as fout:
            df.to_csv(fout, index=False)

    def output(self):
        '''
        '''
        return luigi.LocalTarget(
            os.path.join(self.output_folder, self.output_fname))
