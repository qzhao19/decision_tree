import numpy as np

from ._node import Node

class Tree(object):
    def __init__(self, num_outputs, num_classes, num_features):
        self.num_outputs = num_outputs
        self.num_classes = num_classes
        self.num_features = num_features

        self.max_depth = 0
        self.node_count = 0

        self.nodes = np.array(Node)

    def add_node(self, 
                 depth, 
                 root_indice, 
                 feature_indice, 
                 is_left, 
                 has_missing_value, 
                 threshold, 
                 histogram, 
                 impurity, 
                 improvement):

        # children IDs are set when the child nodes are added
        cur_node = Node(0, 0, 
                        feature_indice, 
                        has_missing_value, 
                        threshold, 
                        histogram, 
                        impurity,
                        improvement)

        self.nodes = np.append(self.nodes, cur_node)

        node_indice = self.node_count + 1
        if depth > 0:
            if is_left:
                self.nodes[root_indice].left_child = node_indice
            else:
                self.nodes[root_indice].right_child = node_indice

        if depth > self.max_depth:
            self.max_depth = depth
        
        return node_indice

    def calculate_feature_importances(self):
        importances = np.zeros(self.num_features)

        if (self.node_count == 0):
            return None
        
        # loop all nodes
        for indice in self.node_count:
            # because leaf node did not have any children 
            # so we only need to test one of the children
            if self.nodes[indice].left_child:
                importances[self.nodes[indice].feature_indice] += self.nodes[indice].improvement
        
        # Normalization
        norm_coeff = 0.0
        for i in range(self.num_features):
            norm_coeff += importances[i]

        if norm_coeff > 0.0:
            for i in range(self.num_features):
                importances[i] = importances[i] / norm_coeff
        
        return importances