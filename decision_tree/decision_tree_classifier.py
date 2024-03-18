import numbers
import numpy as np
import six

from ._builder import DepthFirstTreeBuilder
from ._criterion import Gini
from ._splitter import Splitter
from ._tree import Tree
from ._utils import check_random_state, check_sample_weight, check_input_X_y

class DecisionTreeClassifier(object):
    """A decision tree classifier.

    Parameters:
    ----------
    
    """
    def __init__(self, 
                 criterion_option = "gini", 
                 split_policy = "best", 
                 class_weights = None,
                 max_depth = 4, 
                 min_samples_split = 2,
                 min_samples_leaf = 1,
                 min_weight_fraction_leaf = 0.0, 
                 max_num_features = None,
                 random_state = None):
        
        self.criterion_option = criterion_option
        self.split_policy = split_policy
        self.class_weights = class_weights
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_num_features = max_num_features
        self.random_state = random_state

    def fit(self, X, y):
        random_state = check_random_state(self.random_state)

        num_samples, num_features = X.shape
        X, y = check_input_X_y(X, y)

        # get outputs setting 
        classes = []
        num_classes_list = []

        # reshape y shape (150, ) to (150, 1)
        y_encoded = np.reshape(y, (-1, 1))
        num_outputs = y_encoded.shape[1]
        for k in range(num_outputs):
            classes_k, _ = np.unique(y_encoded[:, k], return_inverse=True)
            classes.append(classes_k)
            num_classes_list.append(classes_k.shape[0])
        self.max_num_classes = np.max(num_classes_list)
        
        # check max_depth
        max_depth = np.iinfo(np.int32).max if self.max_depth is None else self.max_depth

        # check min_samples_leaf
        if isinstance(self.min_samples_leaf, numbers.Integral):
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            min_samples_leaf = int(np.ceil(self.min_samples_leaf * num_samples))

        # check min_samples_split
        if isinstance(self.min_samples_split, numbers.Integral):
            min_samples_split = self.min_samples_split
        else:  # float
            min_samples_split = int(np.ceil(self.min_samples_split * num_samples))
            min_samples_split = max(2, min_samples_split)

        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        # check max_num_features
        if self.max_num_features is None:
            max_num_features = num_features
        elif isinstance(self.max_num_features, numbers.Integral):
            max_num_features = self.max_num_features
        elif isinstance(self.max_num_features, float):
            if self.max_num_features > 0.0:
                max_num_features = max(1, int(self.max_num_features * num_features))
            else:
                max_num_features = 0
        elif isinstance(self.max_num_features, six.string_types):
            if self.max_num_features == "sqrt":
                max_num_features = np.sqrt(num_features)
            elif self.max_num_features == "log2":
                max_num_features = np.log2(num_features)

        # check class weights
        class_weights = check_sample_weight(self.class_weights, num_samples)

        # check min_weight_leaf
        if self.class_weights is None:
            min_weight_leaf = self.min_weight_fraction_leaf * num_samples
        else:
            min_weight_leaf = self.min_weight_fraction_leaf * np.sum(self.class_weights)
        
        # print(X.shape)
        # print(y.shape)

        if self.criterion_option == "gini":
            criterion = Gini(num_outputs, 
                             num_samples, 
                             self.max_num_classes, 
                             num_classes_list,
                             class_weights)
        else:
            NotImplementedError

        splitter = Splitter(criterion, 
                            num_samples,
                            num_features,
                            max_num_features,
                            self.split_policy, 
                            random_state)

        self.tree = Tree(num_outputs,  
                        num_features,
                        self.max_num_classes,
                        num_classes_list)

        builder = DepthFirstTreeBuilder(self.tree,
                                        splitter,
                                        min_samples_split,
                                        min_samples_leaf,
                                        min_weight_leaf,
                                        max_depth,
                                        class_weights)

        builder.build(X, y, num_samples)

        for node in builder.tree.nodes:
            print("left_child = %d, right_child = %d, threshold = %f, improvement = %f, f_indice = %d" 
                  %(node.left_child, node.right_child, node.threshold, node.improvement, node.feature_indice))

    def predict_proba(self, X):
        """predict class probabilities of the input samples X
        """
        num_samples, num_features = X.shape
        X = X.reshape(-1)

        y_proba = self.tree.predict_proba(X, num_samples)

        return y_proba.reshape((num_samples, self.max_num_classes))