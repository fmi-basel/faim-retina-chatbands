'''sampling used for preparing the training data.
'''
import numpy as np


class PlaneSampler:
    '''samples annotated planes from a given stack & annotation.

    '''

    def __init__(self, stack, annotation, n_samples, patch_size):
        '''initializes sampler given an image stack and its annotations.
        '''
        for shape in annotation.shape:
            for ii, jj in zip(stack.shape, shape):
                assert ii == jj
            assert len(stack.shape) == len(shape) + 2
        self.stack = stack
        self.annotation = annotation
        self.n_samples = n_samples
        self.patch_size = patch_size

    def __iter__(self):
        '''iterate over sampled planes with annotation.
        '''
        mask = self.annotation.is_valid().sum(axis=1)
        threshold = 0.25 * self.stack.shape[1]
        indices = np.argwhere(mask >= threshold).flatten()

        # decrease the threshold until we have at least 50 planes.
        while len(indices) <= 50:
            threshold = 0.8 * threshold
            indices = np.argwhere(mask >= threshold).flatten()

        if len(indices) < self.n_samples:
            replace = True
        else:
            replace = False

        for idx in np.random.choice(a=indices,
                                    size=self.n_samples,
                                    #p=probs,
                                    replace=replace):
            yield self.stack[idx], idx, self.annotation[idx]

    def __len__(self):
        '''
        '''
        return self.n_samples
