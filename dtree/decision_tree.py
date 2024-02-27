import numbers
import numpy as np

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
                 criterion_option, 
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
        num_classes_max = np.max(num_classes_list)

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
        elif isinstance(self.max_features, numbers.Integral):
            max_num_features = self.max_features
        else:
            if self.max_num_features > 0.0:
                max_num_features = max(1, int(self.max_num_features * num_features))
            else:
                max_num_features = 0

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
            criterion = Gini(
                num_outputs, 
                num_samples, 
                num_classes_max, 
                num_classes_list,
                class_weights
            )
        else:
            NotImplementedError

        splitter = Splitter(
            criterion, 
            num_samples,
            num_features,
            max_num_features,
            self.split_policy, 
            random_state,
        )

        tree = Tree(num_outputs, num_classes_max, num_features)

        builder = DepthFirstTreeBuilder(
            tree,
            splitter,
            min_samples_split,
            min_samples_leaf,
            min_weight_leaf,
            max_depth,
            class_weights
        )

        builder.build(X, y, num_samples)


        for node in builder.tree.nodes:
            print("left_child = %d, right_child = %d" %(node.left_child, node.right_child))
