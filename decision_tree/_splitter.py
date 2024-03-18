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

    def _random_split_feature(self, X, y, 
                            sample_indices, 
                            feature_indice, 
                            # partition_indice, 
                            partition_threshold, 
                            improvement, 
                            has_missing_value):
        NotImplementedError

    def _best_split_feature(self, X, y, 
                            sample_indices, 
                            feature_indice, 
                            # partition_indice, 
                            partition_threshold, 
                            improvement, 
                            has_missing_value): 
        
        # deep copy sample_indices
        sample_indices = copy.deepcopy(sample_indices)
        # Copy X_feat=X[sample_indices[start:end], feature_indice] training data X for the current node.
        num_samples = self.end - self.start
        f_X = np.zeros(num_samples)
        for i in range(0, num_samples):
            f_X[i] = X[int(sample_indices[i] * self.num_features + feature_indice)]
        
        # check the missing value and shift missing value sample indices to the left
        missing_value_indice = 0
        for i in range(num_samples):
            if np.isnan(f_X[i]):
                f_X[i], f_X[missing_value_indice] = f_X[missing_value_indice], f_X[i]
                sample_indices[i], sample_indices[missing_value_indice] = sample_indices[missing_value_indice], sample_indices[i]
                missing_value_indice += 1
        
        # if all samples have missing value, cannot split feature 
        if missing_value_indice == num_samples:
            return
        
        if missing_value_indice == 0:
            has_missing_value = -1

        # ---Split just based on missing values---
        if missing_value_indice > 0:
            raise NotImplementedError
        
        # ---Split based on threshold---
        # check constant feature in the range of [missing_value_indice : num_samples]
        # we start to check constant feature from non-missing value
        fx_min = fx_max = f_X[missing_value_indice]
        for i in range(missing_value_indice + 1, num_samples):
            if f_X[i] > fx_max:
                fx_max = f_X[i]
            elif f_X[i] < fx_min:
                fx_min = f_X[i]
        
        # not constant feature
        if fx_min + EPSILON < fx_max:
            if missing_value_indice == 0:
                # init class histogram 
                self.criterion.init_children_histogram()
            elif missing_value_indice > 0:
                raise NotImplementedError

            # sort X_feat and sample_indices by X_feat, 
            # leaving missing value at the beginning,
            # samples indices are ordered by their faeture values
            f_X, sample_indices = sort(f_X, sample_indices, missing_value_indice, num_samples, reverse = False)

            # find threshold
            # init next_indice and indice for the position of last and next potentiel split position
            indice = missing_value_indice
            next_indice = missing_value_indice

            # loop samples
            max_improvement = improvement
            max_partition_threshold = partition_threshold
            max_partition_indice = missing_value_indice

            while next_indice < num_samples:
                # if remaining X_feat are constant, stop
                # already sorted f_X[missing_value_indice, num_samples]
                if f_X[next_indice] + EPSILON >= f_X[num_samples - 1]:
                    break
                
                # skip constant X_feat value
                while (next_indice + 1 < num_samples) and (f_X[next_indice] + EPSILON >= f_X[next_indice + 1]):
                    next_indice += 1

                # set next_indice to next position
                next_indice += 1

                # update class histograms from current indice to the new indice (correspond to threshold)
                self.criterion.update_children_histogram(y, sample_indices, next_indice)

                # compute impurity for all outputs of samples for left child and right child
                self.criterion.compute_children_impurity()

                # compute impurity improvement
                improvement_threshold = 0.0
                if missing_value_indice == 0:
                    improvement_threshold = self.criterion.compute_impurity_improvement()
                    print("improvement_threshold = %f" % improvement_threshold)
                elif missing_value_indice > 0:
                    NotImplementedError
                
                if missing_value_indice > 0:
                    raise NotImplementedError
                
                # Identify maximum impurity improvement
                if improvement_threshold > max_improvement:
                    max_improvement = improvement_threshold
                    max_partition_threshold = (f_X[indice] + f_X[next_indice]) / 2.0
                    max_partition_indice = self.start + next_indice
                
                # if right node impurity is 0.0 stop
                if self.criterion.get_right_impurity < EPSILON:
                    break

                indice = next_indice

            if missing_value_indice == 0:
                return sample_indices, max_partition_threshold, max_partition_indice, max_improvement, has_missing_value

            if missing_value_indice > 0:
                raise NotImplementedError


    def split_node(self, X, y, 
                #    feature_indice,
                #    threshold, 
                #    partition_indice,
                #    improvement, 
                #    has_missing_value
                   ):
        # Copy sample_indices = self.sample_indices[start:end]
        # lookup-table to the training data X, y
        f_sample_indices = self.sample_indices[self.start : self.end]

        # -- K random select features --
        # Features are sampled with replacement using the 
        # modern version Fischer-Yates shuffle algorithm

        f_indices = np.arange(0, self.num_features)
        # i = n, instead of n - 1
        i = self.num_features
        while ((i > (self.num_features - self.max_num_features)) or (improvement < EPSILON and i > 0)):
            # increment +=1
            # print("Increment = %s" % increment)
            print("current indice i = %s" % str(i))
            j = 0
            # uniform_int(low, high), low is inclusive and high is exclusive
            if (i > 1):
                j = self.random_state.randint(0, i)
            print("random indice j = %d" % j)
            i -= 1
            f_indices[i], f_indices[j] = f_indices[j], f_indices[i]
            f_indice = f_indices[i]
        
            # split features
            f_has_missing_value = -1
            f_threshold = np.nan
            f_partition_indice = 0
            f_improvement = improvement

            if self.split_policy == "best":
                # sample_indices, max_partition_threshold, max_partition_indice, max_improvement, has_missing_value
                sample_indices, threshold, \
                    partition_indice, improvement, \
                        has_missing_value = self._best_split(X, y, 
                                                             f_sample_indices,
                                                             f_indice, 
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
                self.sample_indices[self.start : self.end] = sample_indices

                split_node_info["feature_indice"] = f_indice
                split_node_info["threshold"] = f_threshold
                split_node_info["partition_indice"] = f_partition_indice
                split_node_info["improvement"] = f_improvement
                split_node_info["has_missing_value"] = f_has_missing_value

