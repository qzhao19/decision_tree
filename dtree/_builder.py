import numpy as np

from ._splitter import Splitter
from ._tree import Tree


class DepthFirstTreeBuilder(object):
    def __init__(self, 
                 num_outputs, 
                 num_samples, 
                 num_features,
                 num_classes_max, 
                 num_classes_list,
                 max_num_features,
                 max_num_thresholds,
                 max_depths,
                 class_weights
                ):
        self.num_outputs = num_outputs
        self.num_samples = num_samples
        self.num_features = num_features
        self.num_classes_max = num_classes_max
        self.num_classes_list = num_classes_list
        self.max_num_features = max_num_features
        self.max_num_thresholds = max_num_thresholds
        self.max_depths = max_depths
        self.class_weights = class_weights


    def build(self, tree, X, y, num_samples):
        
        # node information stack
        # [start, end, depth, parent_indice, is_left]
        node_info_stk = [(0, num_samples, 0, 0, False)]

        

