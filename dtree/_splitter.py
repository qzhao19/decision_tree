import copy
import numpy as np

from ._utils import sort, check_random_state

EPSILON = np.finfo('double').eps


class Splitter(object):
    """Splitter to find the best split for a node
    """
    def __init__(self, 
                 criterion, 
                 num_samples,
                 num_features,
                 max_num_features,
                 split_policy, 
                 random_state):
        
        self.criterion = criterion
        self.num_features = num_features
        self.num_samples = num_samples
        self.max_num_features = max_num_features
        self.split_policy = split_policy
        self.random_state = check_random_state(seed=random_state)

        # initialize sample_indices[0, num_samples] to the training data X, y
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
                self.criterion.init_children_histogram()
            elif missing_value_indice > 0:
                raise NotImplementedError

            # sort X_feat and sample_indices by X_feat, 
            # leaving missing value at the beginning,
            # samples indices are ordered by their faeture values
            X_feat, sample_indices = sort(X_feat, sample_indices, missing_value_indice, num_samples, reverse = False)
            # find threshold
            # init next_indice and indice for the position of last and next potentiel split position
            indice = missing_value_indice
            next_indice = missing_value_indice

            max_improvement = improvement
            max_threshold = threshold
            max_threshold_indice = missing_value_indice

            while next_indice < num_samples:
                # if remaining X_feat are constant, stop
                if X_feat[next_indice] + EPSILON >= X_feat[num_samples - 1]:
                    break

                # skip constant X_feat value
                while (next_indice + 1 < num_samples) and (X_feat[next_indice] + EPSILON >= X_feat[next_indice + 1]):
                    next_indice += 1
                
                # increment next_indice
                next_indice += 1

                # update class histograms from current indice to the new indice
                self.criterion.update_children_histogram(y, sample_indices, next_indice)

                # compute impurity for all outputs of samples for left child and right child
                self.criterion.compute_children_impurity()

                # compute impurity improvement
                threshold_improvement = 0.0
                if missing_value_indice == 0:
                    threshold_improvement = self.criterion.compute_impurity_improvement()
                    print("threshold_improvement = %f" % threshold_improvement)
                elif missing_value_indice > 0:
                    NotImplementedError
                
                if missing_value_indice > 0:
                    raise NotImplementedError

                # Identify maximum impurity improvement
                if threshold_improvement > max_improvement:
                    max_improvement = threshold_improvement
                    max_threshold = (X_feat[indice] + X_feat[next_indice]) / 2.0
                    max_threshold_indice = self.start + next_indice
                    print("current position = %d, value = %f, next position = %d, value = %f" 
                          % (indice, X_feat[indice], next_indice, X_feat[next_indice]))

                # if right node impurity is 0.0 stop
                if self.criterion.get_right_impurity < EPSILON:
                    break

                indice = next_indice

            if missing_value_indice == 0:
                return sample_indices, max_threshold, max_threshold_indice, max_improvement, has_missing_value

            if missing_value_indice > 0:
                raise NotImplementedError
            
        # return split_info
                
    def _random_split(self):
        pass
    
    def split_node(self, X, y, split_node_info):
        # Copy sample_indices = self.sample_indices[start:end]
        # lookup-table to the training data X, y
        f_sample_indices = self.sample_indices[self.start : self.end]
        
        # -- K random select features --
        # Features are sampled with replacement using the 
        # modern version Fischer-Yates shuffle algorithm

        feat_indices = np.arange(0, self.num_features)
        # improvement = 0.0
        # i = n, instead of n - 1
        i = self.num_features
        # print("current indice1 i = %s" % str(i))
        increment = 0
        while ((i > (self.num_features - self.max_num_features)) or (split_node_info["improvement"] < EPSILON and i > 0)):
            increment +=1
            print("Increment = %s" % increment)
            print("current indice i = %s" % str(i))
            j = 0
            # uniform_int(low, high), low is inclusive and high is exclusive
            if (i > 1):
                j = self.random_state.randint(0, i)
            print("random indice j = %d" % j)
            i -= 1
            feat_indices[i], feat_indices[j] = feat_indices[j], feat_indices[i]
            feat_indice = feat_indices[i]

            # split features
            f_has_missing_value = -1
            f_threshold = 0.0
            f_partition_indice = 0
            f_improvement = split_node_info["improvement"]

            if self.split_policy == "best":
                sample_indices, threshold, partition_indice, improvement, has_missing_value = self._best_split(X, y, 
                                          sample_indices, 
                                          feat_indice, 
                                          f_threshold, 
                                          f_partition_indice, 
                                          f_improvement, 
                                          f_has_missing_value)

                f_improvement = improvement
                f_has_missing_value = has_missing_value
                f_threshold = threshold
                f_partition_indice = partition_indice
            else:
                raise NotImplementedError

            if f_improvement > improvement:
                self.sample_indices[self.start : self.end] = split_node_info["sample_indices"]

                split_node_info["feature_indice"] = feat_indice
                split_node_info["threshold"] = f_threshold
                split_node_info["partition_indice"] = f_partition_indice
                split_node_info["improvement"] = f_improvement
                split_node_info["has_missing_value"] = f_has_missing_value
                

