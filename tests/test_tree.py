import unittest
import numpy as np

from context import dtree
from dtree._criterion import Gini
from dtree._tree import Tree


class TestTree(unittest.TestCase):
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
    sample_indices = np.arange(0, n_samples)
    n_features = 4
    criterion = Gini(n_outputs, n_samples, n_classes_max, n_classes, class_weight)
    tree = Tree(n_outputs, n_classes_max, n_features)

    def test_add_node(self):
        # split node if it is not leaf node
        feature_indice = 0
        has_missing_value = -1  # default: no missing value
        threshold = 0.0
        improvement = 0.0

        self.criterion.compute_node_histogram(self.y, self.sample_indices, 0, self.n_samples)
        weighted_histogram = self.criterion.get_weighted_histogram
        self.criterion.compute_node_impurity()
        impurity = self.criterion.get_impurity_node
        node_indice = self.tree.add_node(
                0, 
                0, 
                False,
                feature_indice, 
                has_missing_value, 
                threshold, 
                weighted_histogram, 
                impurity, 
                improvement,
            )
        self.assertTrue((node_indice == 0))

if __name__ == '__main__':
    unittest.main()