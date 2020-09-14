import numpy as np


class PlaneSampler:
    '''
    '''

    def __init__(self, stack, annotation, n_samples, patch_size):
        '''
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
        '''
        '''
        mask = self.annotation.is_valid().sum(axis=1)
        threshold = 0.25 * self.stack.shape[1]
        indices = np.argwhere(mask >= threshold).flatten()

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
