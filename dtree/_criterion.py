import numpy as np

class Gini(object):
    def __init__(self, num_outputs, num_classes, num_samples, class_weight):
        self.num_outputs = num_outputs
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.class_weight = class_weight

    def _compute_node_histogram(self, y, samples, start, end):
        
        self.node_weighted_histogram = np.zeros((self.num_outputs, self.num_classes))
        self.node_weighted_num_samples = np.zeros((self.num_outputs))
        for o in range(self.num_outputs):
            histogram = {}
            for i in range(start, end):
                histogram[y[samples[i] * self.num_outputs + o]] += 1

            for c in range(self.num_classes):
                weighted_count = self.class_weight[o * self.num_classes + c] * histogram[c]
                self.node_weighted_histogram[o][c] = weighted_count
                self.node_weighted_num_samples[o] += weighted_count

    
    def _compute_impurity(self, histogram):
        sum_count = 0
        sum_count_squared = 0
        