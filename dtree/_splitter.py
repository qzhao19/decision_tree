import copy
import numpy as np

from ._utils import sort

INFINITY = np.inf
EPSILON = np.finfo('double').eps


class Splitter(object):
    """Splitter to find the best split for a node
    """
    def __init__(self, 
                 criterion, 
                 num_samples,
                 num_features,
                 max_num_features,
                 split_policy = "best", 
                 random_state = None):
        
        self.criterion = criterion
        self.num_features = num_features
        self.num_samples = num_samples
        self.max_num_features = max_num_features
        self.split_policy = split_policy
        self.random_state = random_state

        self.sample_indices = np.arange(0, num_samples)

    def init_node(self, y, start, end):
        """Initialize node and calculate weighted histograms 
        for all outputs and impurity for the node.
        """
        self.start = start
        self.end = end

        self.criterion.compute_node_histogram(y, self.sample_indices, self.start, self.end)
        self.criterion.compute_node_impurity()

    def _best_split(self, X, y, 
                    sample_indices,
                    feature_indice,
                    threshold, 
                    partition_indice, 
                    improvement, 
                    has_missing_value):

        # Copy X_feat=X[sample_indices[start:end], feature_indice] training data X for the current node.
        num_samples = self.end - self.start
        X_feat = np.zeros(num_samples)
        for i in range(num_samples):
            X_feat[i] = X[int(sample_indices[i] * self.num_features + feature_indice)]
        
        # check the missing value and move them to the beginning of 
        missing_value_indice = 0
        for i in range(num_samples):
            if np.isnan(X_feat[i]):
                X_feat[i], X_feat[missing_value_indice] = X_feat[missing_value_indice], X_feat[i]
                sample_indices[i], sample_indices[missing_value_indice] = sample_indices[missing_value_indice], sample_indices[i]
                missing_value_indice += 1
        
        # if all samples have missing value, cannot split feature 
        if missing_value_indice == num_samples:
            return

        # ---Split just based on missing values---
        if missing_value_indice > 0:
            raise NotImplementedError
        
        # ---Split based on threshold---
        # check constant feature in the range of [missing_value_indice:num_samples]
        feat_max = feat_min = X_feat[missing_value_indice]
        for i in range(missing_value_indice+1, num_samples):
            if X_feat[i] > feat_max:
                feat_max = X_feat[i]
            elif X_feat[i] < feat_min:
                feat_min = X_feat[i]
        
        # not constant feature
        if feat_min + EPSILON < feat_max:
            if missing_value_indice == 0:
                # init class histogram 
                self.criterion.init_threshold_histogram()
            elif missing_value_indice > 0:
                raise NotImplementedError

            # sort X_feat and sample_indices by X_feat, 
            # leaving missing value at the beginning,
            # samples indices are ordered by their faeture values
            X_feat, sample_indices = sort(X_feat, sample_indices, self.start, self.end)