'''very basic utility functions. We avoid module level imports
in order to avoid dependencies if not required.
'''


def set_seeds(seed_val=42):
    '''fix seeds for reproducibility.

    '''
    from numpy.random import seed
    from tensorflow import set_random_seed
    seed(seed_val)
    set_random_seed(seed_val)


def configure_gpu_session(fixed_seed=None, n_threads=None):
    '''configures default tf session. This seems to be necessary with
    shared, virtual gpus.

    '''
    import tensorflow as tf
    from keras import backend

    if fixed_seed is not None:
        set_seeds(fixed_seed)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if n_threads is not None:
        config.intra_op_parallelism_threads = n_threads
        config.inter_op_parallelism_threads = n_threads
    backend.set_session(tf.Session(config=config))


def get_zero_based_task_id(default_return=None):
    '''fetches the environment variable for this process' task id.

    Returns None if process is not run in an SGE environment.

    '''
    import os
    sge_id = os.environ.get('SGE_TASK_ID', None)
    if sge_id is None:
        return default_return

    return int(sge_id) - 1
