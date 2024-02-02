import unittest

from ..dtree._criterion import Gini

def my_sum(a,b):
    return a+b

class my_test(unittest.TestCase):
    def test1_oo1(self):
        print(my_sum(5,3))

    def test1_002(self):
        print(my_sum(2,8))


