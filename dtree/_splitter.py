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
        
        # y is not constant (impurity > 0)
        # has been checked by impurity stop criteria in build()
        # moving on we can assume at least 2 samples

        # Copy f_X=X[samples[start:end],f] training data X for the current node.
        num_samples = self.end - self.start
        f_x = np.zeros(num_samples)
        for i in range(num_samples):
            f_x[i] = X[samples[i] * self.num_features + feature_indice]

        # Detect samples with missing values and 
        # move them to the beginning of the samples vector
        missing_value_indice = 0
        for i in range(num_samples):
            if np.isnan(f_x[i]):
                f_x[i], f_x[missing_value_indice] = f_x[missing_value_indice], f_x[i]
                samples[i], samples[missing_value_indice] = f_x[missing_value_indice], f_x[i]
                missing_value_indice += 1
        
        # Can not split feature when all values are NA
        if missing_value_indice == num_samples:
            return 
        
        

