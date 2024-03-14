import unittest
import numpy as np

from context import decision_tree
from decision_tree._criterion import Gini
from decision_tree._tree import Tree
from decision_tree._builder import DepthFirstTreeBuilder
from decision_tree._splitter import Splitter
from decision_tree._utils import check_sample_weight


class TestBuilder(unittest.TestCase):
    def calculate_n_classes(classes):
        n_classes = []
        for o in range(len(classes)):
            n_classes.append(len(classes[o]))
        
        return np.asarray(n_classes)

    X = np.array([5.2, 3.3, 1.2, 0.3,
            4.8, 3.1 , 1.6, 0.2,
            4.75, 3.1, 1.32, 0.1,
            5.9, 2.6, 4.1 , 1.2,
            5.1 , 2.2, 3.3, 1.1,
            5.2, 2.7, 4.1, 1.3,
            6.6, 3.1 , 5.25, 2.2,
            6.3, 2.5, 5.1 , 2.,
            6.5, 3.1 , 5.2, 2.1])
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    classes = np.array([["setosa", "versicolor", "virginica"]])
    n_classes = calculate_n_classes(classes)
    n_outputs = len(classes)
    n_classes_max = np.max(n_classes)
    class_weight = np.ones((n_outputs * n_classes_max))
    n_samples = int(len(y) / n_outputs)
    n_features = 4

    sample_indices = np.arange(0, n_samples)
    n_features = 4

    tree = Tree(n_outputs, n_features, n_classes_max, n_classes)
    criterion = Gini(n_outputs, n_samples, n_classes_max, n_classes, class_weight)
    splitter = Splitter(criterion, n_samples, n_features, n_features, split_policy = "best", random_state = None)

    max_depth = 4
    min_samples_split = 2
    min_samples_leaf = 1
    min_weight_fraction_leaf = 0.0
    min_weight_leaf = min_weight_fraction_leaf * n_samples
    class_weights = check_sample_weight(None, n_samples)

    def test_builder(self):
        builder = DepthFirstTreeBuilder(self.tree, 
                                        self.splitter, 
                                        self.min_samples_split, 
                                        self.min_samples_leaf, 
                                        self.min_weight_leaf, 
                                        self.max_depth, 
                                        self.class_weights)

        builder.build(self.X, self.y, self.n_samples)

        for node in builder.tree.nodes:
            print("left_child = %d, right_child = %d, threshold = %f, improvement = %f" 
                  %(node.left_child, node.right_child, node.threshold, node.improvement))

if __name__ == '__main__':
    unittest.main()
