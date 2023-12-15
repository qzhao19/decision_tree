import numpy as np

class FeatureSplitter(object):
    def __init__(self, 
                 criterion, 
                 num_samples, 
                 num_features, 
                 num_outputs, 
                 num_classes,
                 num_rand_thresholds, 
                 class_weight):
        
        self.criterion = criterion
        self.num_samples = num_samples
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.num_classes = num_classes
        self.num_rand_thresholds = num_rand_thresholds
        self.class_weight = class_weight

        self.samples = np.zeros((num_samples))

    # Initialize node and calculate weighted histograms for all outputs and impurity for the node.
    def init_node(self, y, start, end):
        self.start = start
        self.end = end

        self.criterion._compute_node_histogram(y, self.samples, self.start, self.end)

        self.criterion._compute_node_impurity()


    def _best_split_feature(self, 
                           X, y, 
                           samples, 
                           feature_indice,  
                           threshold, 
                           partition_indice, 
                           improvement, 
                           missing_value):
        
        




