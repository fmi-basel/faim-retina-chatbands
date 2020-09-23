'''callbacks for training.
'''
import numpy as np

from keras.callbacks import TensorBoard
import keras.backend as backend


def CosineAnnealingSchedule(lr_max, lr_min, epoch_max, reset_decay=1):
    '''create a learning rate scheduler that follows the approach from

    SGDR: Stochastic gradient descent with warm restarts,
    Loshchilov & Hutter, ICLR 2017

    TODO implement increasing epoch_max multiplier.
    TODO implement checkpointing the model each time a reset is done.

    '''
    if lr_max <= lr_min:
        raise ValueError(
            'lr_max has to be larger than lr_min! {} !> {}'.format(
                lr_max, lr_min))

    def schedule(epoch, current_lr=None):
        '''schedule function to be passed to LearningRateScheduler.

        '''
        current_lr_max = reset_decay**-int(epoch // epoch_max) * lr_max
        cosine_factor = (1 +
                         np.cos(float(epoch % epoch_max) / epoch_max * np.pi))
        return lr_min + 0.5 * (current_lr_max - lr_min) * cosine_factor

    return schedule


class ExtendedTensorBoard(TensorBoard):
    '''adds learning rate to logged metrics.

    NOTE this becomes obsolete with newer versions of tf.
    '''

    def __init__(self, *args, **kwargs):
        '''
        '''
        super().__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        '''
        '''
        logs.update({'lr': backend.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)
