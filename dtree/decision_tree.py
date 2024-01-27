import numpy as np

from ._builder import DepthFirstTreeBuilder
from ._criterion import Gini
from ._splitter import Splitter
from ._utils import check_random_state

class DecisionTreeClassifier(object):
    """A decision tree classifier.

    Parameters:
    ----------
    
    """
    def __init__(self, 
                 criterion_option, 
                 split_strategy, 
                 class_weights,
                 max_depth, 
                 min_samples_split,
                 min_samples_leaf,
                 min_weight_fraction_leaf, 
                 max_num_features,
                 random_state
                 ):
        
        self.criterion_option = criterion_option
        self.split_strategy = split_strategy
        self.class_weights = class_weights
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_num_features = max_num_features
        self.random_state = random_state

    def fit(self, X, y):

        check_random_state(self.random_state)

        num_samples, num_features = X.shape

        # get outputs setting 
        y = np.atleast_1d(y)
        # reshape y shape (150, ) to (150, 1)
        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        self.num_outputs = y.shape[1]

        self.classes = []
        self.num_classes_list = []

        # if self.class_weights is not None:
        #     y_original = np.copy(y)

        y_encoded = np.zeros(y.shape, dtype=int)
        for k in range(self.n_outputs_):
            classes_k, y_encoded[:, k] = np.unique(y[:, k], return_inverse=True)
            self.classes.append(classes_k)
            self.num_classes_list.append(classes_k.shape[0])









