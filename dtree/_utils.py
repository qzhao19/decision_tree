import numbers
import numpy as np

def sort(x, y, start, end, reverse=True):
    x_y = [(x[i], y[i]) for i in range(len(x))]
    
    if not reverse:
        x_y[start:end] = sorted(x_y[start:end], key=lambda z: z[0])
    else:
        x_y[start:end] = sorted(x_y[start:end], key=lambda z: z[0], reverse=True)
    
    x[:] = [z[0] for z in x_y]
    y[:] = [z[1] for z in x_y]

    return np.array(x), np.array(y)

# from sklearn
def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    :class:`numpy:numpy.random.RandomState`
        The random state object based on `seed` parameter.

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )


def check_sample_weight(sample_weight, num_samples, dtype=None):
    """Validate sample weights.
    """

    if dtype is not None and dtype not in [np.float32, np.float64]:
        dtype = np.float64

    if sample_weight is None:
        sample_weight = np.ones(num_samples, dtype=dtype)
    elif isinstance(sample_weight, numbers.Number):
        sample_weight = np.full(num_samples, sample_weight, dtype=dtype)
    
    return sample_weight


def check_input_X_y(X, y):
    if X.size() != y.size():
        raise ValueError("X and y must be the same size.")
    
    if X.ndim != 0:
        raise ValueError("X expected 2D array, but got scalar array.")

    if X.ndim != 1:
        X = X.reshape(-1)

    y = np.atleast_1d(y)
    return X, y
