import numbers
import numpy as np

def sort(x, y, start, end, reverse=False):
    x_y = [(x[i], y[i]) for i in range(len(x))]
    
    if not reverse:
        x_y[start:end] = sorted(x_y[start:end], key=lambda z: z[0])
    else:
        x_y[start:end] = sorted(x_y[start:end], key=lambda z: z[0], reverse=True)
    
    sorted_x = [z[0] for z in x_y]
    sorted_y = [z[1] for z in x_y]

    return np.array(sorted_x), np.array(sorted_y)

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

    Source: https://github.com/scikit-learn/scikit-learn/blob/6bb2762b0b8c12b2e2578ef528ea7ea30580f9fa/sklearn/utils/validation.py#L1375
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    elif isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    elif isinstance(seed, np.random.RandomState):
        return seed
    else:
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
    if (not isinstance(X, np.ndarray)) or (not isinstance(y, np.ndarray)):
        raise ValueError("input should be in np.ndarray format, got %s"
                         % type(X))
    
    if len(X) != len(y):
        raise ValueError("X and y must be the same size.")
    
    if X.ndim != 1:
        X = X.reshape(-1)

    y = np.atleast_1d(y)
    return X.astype(np.double), y.astype(int)
