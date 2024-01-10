import copy
import numpy as np

from ._utils import sort

PRECISION = 1e-7

class FeatureSplitter(object):
    def __init__(self, 
                 criterion, 
                 num_samples, 
                 num_features, 
                 num_classes,
                 max_num_features,
                 max_num_thresholds, 
                 class_weight):
        
        self.criterion = criterion
        self.num_samples = num_samples
        self.num_features = num_features
        self.num_classes = num_classes
        self.max_num_features = max_num_features
        self.max_num_thresholds = max_num_thresholds
        self.class_weight = class_weight

        self.samples = np.arange(0, num_samples)

    def init_node(self, y, start, end):
        """Initialize node and calculate weighted histograms 
        for all outputs and impurity for the node.
        """
        self.start = start
        self.end = end

        self.criterion.compute_node_histogram(y, self.samples, self.start, self.end)
        self.criterion.compute_node_impurity()


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

        # Copy X_feat_i=X[samples[start:end],f] training data X for the current node.
        num_samples = self.end - self.start
        X_feat = np.zeros(num_samples)
        for i in range(num_samples):
            X_feat[i] = X[samples[i] * self.num_features + feature_indice]

        # Detect samples with missing values and 
        # move them to the beginning of the samples vector
        missing_indice = 0
        for i in range(num_samples):
            if np.isnan(X_feat[i]):
                X_feat[i], X_feat[missing_indice] = X_feat[missing_indice], X_feat[i]
                samples[i], samples[missing_indice] = X_feat[missing_indice], X_feat[i]
                missing_indice += 1
        
        # Can not split feature when all values are NA
        if missing_indice == num_samples:
            return 
        
        if missing_indice > 0:
            print("NO YET IMPLEMENT")

        # Split based on threshold
        feat_max = feat_min = X_feat[missing_indice]
        for i in range(missing_indice+1, num_samples):
            if X_feat[i] > feat_max:
                feat_max = X_feat[i]
            elif X_feat[i] < feat_min:
                feat_min = X_feat[i]
        

        if feat_min + PRECISION < feat_max:

            if missing_indice == 0:
                self.criterion.init_threshold_histogram()
            elif missing_indice > 0:
                print("NO YET IMPLEMENT")
            
            # Loop: all thresholds
            X_feat, samples = sort(X_feat, samples, missing_indice, num_samples)

            # Find threshold with maximum impurity improvement
            # Initialize position of last and next potential split to number of missing 
            
            prev_indice = missing_indice, next_indice = missing_indice
            max_improvement = 0.0
            max_threshold = 0.0
            max_indice = missing_indice

            while next_indice < num_samples:

                # 
                if X_feat[next_indice] + PRECISION >= X_feat[num_samples - 1]:
                    break
                # Skip constant Xf values
                while (next_indice + 1 < num_samples) and (X_feat[next_indice] + PRECISION >= X_feat[next_indice + 1]):
                    next_indice += 1
                
                next_indice += 1

                # Update class histograms for all outputs for using a threshold on values
                # from current indice p to the new position np (correspond to thresholds)
                self.criterion.update_threshold_histograms(y, samples, next_indice)
                



            
