import numpy as np

class Node(object):
    """Definition of a binary tree node
    
    Parameters:
    ----------
        left_child: int
            left child indice 
        
        right_child: int
            right child indice

        feature_indices: int
            the indice of feature chosen we want to split
        
        has_missing_value: int
            if have the missing value if -1: No missing value, 
            0: in the left child , 1: right child
        
        threshold: float
            the threshold value decide how to split current feature 
        
        histogram: 2d array
            weighted number of samples per class per output
        
        impurity: float
        
        improvement: float
            for the feature importance

    """
    def __init__(self, 
                 left_child, 
                 right_child, 
                 feature_indice, 
                 has_missing_value, 
                 threshold, 
                 histogram, 
                 impurity, 
                 improvement):
            self.left_child = left_child
            self.right_child = right_child
            self.feature_indice = feature_indice
            self.has_missing_value = has_missing_value
            self.threshold = threshold
            self.histogram = histogram
            self.impurity = impurity
            self.improvement = improvement

    

