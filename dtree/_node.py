import numpy as np

class Node(object):
    def __init__(self, 
        left_child, 
        right_child, 
        feature_indice, 
        missing_value, 
        threshold, 
        histogram, 
        impurity, 
        improvement):
            self.left_child = left_child
            self.right_child = right_child
            self.feature_indice = feature_indice
            self.missing_value = missing_value
            self.threshold = threshold
            self.histogram = histogram
            self.impurity = impurity
            self.improvement = improvement

    

