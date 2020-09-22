'''collector for stacks to be processed.
'''
import os
import glob
import itertools
import logging

import luigi
import pandas

from .ioutils import parse_meta


# Typical file extensions of stacks that need to be processed.
DEFAULT_FNAME_PATTERNS = [
        '*confGFP*stk',
        '*conf488*stk',
        '*.czi']


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
