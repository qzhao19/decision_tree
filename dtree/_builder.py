import numpy as np

from ._splitter import EPSILON
from ._tree import Tree


class NodeInfo(object):
    def __init__(self, start, end, depth, parent_indice, is_left):
        self.start = start
        self.end = end
        self.depth = depth
        self.parent_indice = parent_indice
        self.is_left = is_left
        

class DepthFirstTreeBuilder(object):
    """Build a binary decision tree in depth-first order.
    """
    def __init__(self, 
                 tree,
                 splitter,
                 min_samples_split,
                 min_samples_leaf,
                 min_weight_leaf,
                 max_depth,
                 class_weights
                ):
        self.tree = tree
        self.splitter = splitter

        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf

        self.max_depth = max_depth
        self.class_weights = class_weights


    def build(self, X, y, num_samples):
        
        # node information stack
        # node_info_stk = [start, end, depth, parent_indice, is_left]
        node_info_stk = []
        node_info_stk.append(NodeInfo(0, num_samples, 0, 0, False))
        
        while node_info_stk:
            # Pop current node information from the stack
            node_info = node_info_stk.pop()

            # compute the number of samples in the current node
            num_samples_node = node_info.end - node_info.start

            # calculate number of samples per class histogram for all outputs
            # and impurity for the current node
            self.splitter.init_node(y, node_info.start, node_info.end)
            histogram = self.splitter.criterion.get_weighted_histogram
            impurity = self.splitter.criterion.get_impurity_node
            partition_indice = 0

            # stop criterion is met node becomes a leaf node
            is_leaf = (node_info.depth >= self.max_depth or 
                       num_samples_node < self.min_samples_split or 
                       num_samples_node < 2 * self.min_samples_leaf or 
                       num_samples_node < 2 * self.min_weight_leaf)
            
            is_leaf = is_leaf or impurity <= EPSILON

            # split node if it is not leaf node
            feature_indice = 0
            has_missing_value = -1  # default: no missing value
            threshold = None
            improvement = 0.0

            if not is_leaf:
                split_info = self.splitter.split_node(X, y)

                # print(split_info)
                feature_indice = split_info["feature_indice"]
                has_missing_value = split_info["has_missing_value"]
                threshold = split_info["threshold"]
                improvement = split_info["improvement"]
                partition_indice = split_info["partition_indice"]

                if split_info["improvement"] <= EPSILON:
                    is_leaf = True

            node_indice = self.tree.add_node(
                node_info.depth, 
                node_info.parent_indice, 
                node_info.is_left,
                feature_indice, 
                has_missing_value, 
                threshold, 
                histogram, 
                impurity, 
                improvement,
            )

            if not is_leaf:
                # push right child node info into the stack
                node_info_stk.append(NodeInfo(partition_indice, node_info.end, node_info.depth + 1, node_indice, False))

                # push left child node info into the stack
                node_info_stk.append(NodeInfo(node_info.start, partition_indice, node_info.depth + 1, node_indice, True))


