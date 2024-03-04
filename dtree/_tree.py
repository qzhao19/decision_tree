import numpy as np

from ._node import Node

class IndiceInfo(object):
    def __init__(self, indice, weight):
        self.indice = indice
        self.weight = weight


class Tree(object):
    """The binary tree is represented as a number of parallel arrays. The i-th
    element of each array holds information about the node `i`. Node 0 is the
    tree's root. 
    
    For node data stored at index i, the two child nodes are at 
    index (2 * i + 1) and (2 * i + 2); the parent node is (i - 1) // 2  
    (where // indicates integer division).
    """
    def __init__(self, num_outputs, num_classes, num_features):
        self.num_outputs = num_outputs
        self.num_classes = num_classes
        self.num_features = num_features

        self.max_depth = 0
        self.node_count = 0

        self.nodes = []

    def add_node(self, 
                 depth, 
                 parent_indice, 
                 is_left, 
                 feature_indice, 
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
        self.nodes.append(cur_node)
        
        node_indice = self.node_count
        self.node_count += 1

        if depth > 0:
            if is_left:
                self.nodes[parent_indice].left_child = node_indice
            else:
                self.nodes[parent_indice].right_child = node_indice

        if depth > self.max_depth:
            self.max_depth = depth
        
        return node_indice

    def compute_feature_importances(self):
        """compute the importances of each feature 
        """
        importances = np.zeros(self.num_features, dtype=np.double)

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
    
    def predict_proba(self, X):
        """predict classes probabilities
        """
        num_samples = X.shape[0]
        y_proba = np.zeros((num_samples * self.num_outputs * self.num_classes), dtype=np.double)

        for i in range(num_samples):
            node_idx_info_stk = []
            leaf_idx_info_stk = []

            # start from the root to leaf node
            node_idx_info_stk.append(IndiceInfo(0, 1.0))

            while node_idx_info_stk:
                node_idx_info1 = node_idx_info_stk.pop()

                # follow path until leaf node
                # 
                while self.nodes[node_idx_info1.indice].left_child > 0:



