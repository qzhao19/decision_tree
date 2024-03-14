import unittest
import numpy as np

from context import decision_tree
from decision_tree._utils import sort

class TestUtilsFunction(unittest.TestCase):
    def test_sort(self):
        X = np.array([5.2, 3.3, 1.2, 0.3,
                      4.8, 3.1 , 1.6, 0.2, 4.75])
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        num_samples = X.shape[0]
        sorted_X, sorted_y = sort(X, y, 0, num_samples)
        self.assertTrue((sorted_X == np.array([5.2, 4.8, 4.75, 3.3, 3.1, 1.6, 1.2, 0.3, 0.2])).all())
        self.assertTrue((sorted_y == np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])).all())

if __name__ == '__main__':
    unittest.main()