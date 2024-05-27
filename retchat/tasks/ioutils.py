# If your training data has different scaling than your test data, pay attention to _parse_scales below.
import os
import re
import glob
from xml.etree import ElementTree
import logging

import numpy as np
import czifile
from skimage.external import tifffile

#import javabridge
#import bioformats
#javabridge.start_vm(class_path=bioformats.JARS)

logger = logging.getLogger('luigi-interface')


def parse_meta(path):
    '''convenience function to get meta data.
    '''
    logger.debug('Opening file {}'.format(path))
    ext = os.path.splitext(path)[1]
    if ext.lower() == '.czi':
        return CziParser.parse(path)
    elif ext.lower() == '.stk':
        return StkParser.parse(path)
    elif ext.lower() == '.tif':
        return StkParser.parse(path)
    elif ext.lower() == '.tiff':
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
#    elif ext.lower() == '.vsi':
#        img = bioformats.load_image(path, c=1, z=0, t=0, series=None, index=None, rescale=True, wants_max_intensity=False, channel_names=None)
#        logger.warn('Loading only channel 1 from vsi!')
#        return img    
        
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
                raise RuntimeError(
                    'Multiple wavelenghts found for channel {}!'.format(
                        channel.get('Name')))
            if not wavelengths:
                continue
            info['channel_' + channel.get('Name')] = round(
                float(wavelengths[0].text))
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
            scales['scale_z'] = StkParser._find_zstep(StkParser._find_ndfile(path))
        except RuntimeError as err:
            logging.getLogger('luigi-interface').error('{}. Using 0.3 um as z-step size.'.format(err))
            scales['scale_z'] = 3e-7    

        # Pixel spacing based on lookup-table from W1-scope/cameras.
        shape = filehandle.series[0].shape[-2:]
        if np.all(shape == (2048, 2048)):
            in_plane = 0.103 * 1e-6
        elif np.all(shape == (1200, 1200)):
            in_plane = 0.168 * 1e-6
        else:
            in_plane = 0.1 * 1e-6
            #raise RuntimeError('Unknown XY setting for .stk: {}'.format(path))
 
        scales['scale_x'] = in_plane
        scales['scale_y'] = in_plane
	# pseudo pixel spacing to enforce that data is not downsampled
        scales['scale_x'] = 2.07e-7
        scales['scale_y'] = 2.07e-7
        scales['scale_z'] = 3e-7
        print(scales)

        return scales

    @staticmethod
    def _find_zstep(ndfilepath):
        with open(ndfilepath, 'r') as fin:
            print(ndfilepath)
            txt = fin.read()
            return float(re.search('"ZStepSize", (\d+.\d+)',
                                   txt).group(1)) * 1e-6

    @staticmethod
    def _find_ndfile(stkpath):
        '''tries to identify the .nd file for a given .stk
        '''
        dirname, _ = os.path.split(stkpath)
        candidates = glob.glob(os.path.join(dirname, '*nd'))

        if len(candidates) <= 0:
            raise RuntimeError(
                'Could not find any .nd file for {}!'.format(stkpath))

        candidates = sorted(candidates,
                            key=lambda candidate: len(
                                os.path.commonprefix([stkpath, candidate])))

        if len(os.path.commonprefix([stkpath, candidates[-1]
                                     ])) <= len(dirname):
            raise RuntimeError(
                'Could not find matching .nd file for {}!'.format(stkpath))

        return candidates[-1]
