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
    
    criterion = Gini(n_outputs, n_samples, n_classes_max, n_classes, class_weight)

    