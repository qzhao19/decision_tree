import unittest
import numpy as np

from context import decision_tree
from decision_tree._criterion import Gini


class TestCriterionGini(unittest.TestCase):
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
    
    criterion = Gini(n_outputs, n_samples, n_classes_max, n_classes, class_weight)

    def test_compute_node_histogram(self):
        self.criterion.compute_node_histogram(self.y, self.sample_indices, 0, self.n_samples)
        weighted_histogram = self.criterion.get_weighted_histogram
        self.assertTrue((weighted_histogram == np.array([[3., 3., 3.]])).all())

    def test_compute_node_impurity(self):
        self.criterion.compute_node_impurity()
        impurity_node = self.criterion.get_node_impurity
        self.assertTrue((np.round(impurity_node, decimals=6) == 0.666667))

    def test_init_children_histogram(self):
        self.criterion.init_children_histogram()
        weighted_histogram_left = self.criterion.left_weighted_histogram
        weighted_histogram_right = self.criterion.right_weighted_histogram
        weighted_num_samples_left = self.criterion.left_weighted_num_samples
        weighted_num_samples_right = self.criterion.right_weighted_num_samples

        self.assertTrue((weighted_histogram_left == np.array([[0., 0., 0.]])).all())
        self.assertTrue((weighted_histogram_right == np.array([[3., 3., 3.]])).all())
        self.assertTrue((weighted_num_samples_left == np.array([0.])).all())
        self.assertTrue((weighted_num_samples_right == np.array([9.])).all())

    def test_update_children_histogram(self):
        self.criterion.update_children_histogram(self.y, self.sample_indices, 3)
        weighted_histogram_left = self.criterion.left_weighted_histogram
        weighted_histogram_right = self.criterion.right_weighted_histogram
        weighted_num_samples_left = self.criterion.left_weighted_num_samples
        weighted_num_samples_right = self.criterion.right_weighted_num_samples

        self.assertTrue((weighted_histogram_left == np.array([[3., 0., 0.]])).all())
        self.assertTrue((weighted_histogram_right == np.array([[0., 3., 3.]])).all())
        self.assertTrue((weighted_num_samples_left == np.array([3.])).all())
        self.assertTrue((weighted_num_samples_right == np.array([6.])).all())

    def test_get_node_impurity(self):
        impurity = self.criterion.get_node_impurity
        self.assertTrue((np.round(impurity, decimals=6) == 0.666667))

if __name__ == '__main__':
    unittest.main()
