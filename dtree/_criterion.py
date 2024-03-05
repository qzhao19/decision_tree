import numpy as np

class Criterion(object):
    def __init__(self, 
                 num_outputs, 
                 num_samples, 
                 max_num_classes, 
                 num_classes_list,
                 class_weights):
        self.num_outputs = num_outputs
        self.num_samples = num_samples
        self.max_num_classes = max_num_classes
        self.num_classes_list = num_classes_list
        self.class_weights = class_weights

        # impurity of in the current node
        self.node_impurity = np.zeros(num_outputs, dtype=np.double)
        # impurity of in the left node with values smaller than threshold
        self.left_impurity = np.zeros(num_outputs, dtype=np.double)
        # impurity of in the right node with values bigger that threshold
        self.right_impurity = np.zeros(num_outputs, dtype=np.double)

        # weighted number of samples in the node, left child and right child
        self.node_weighted_num_samples = np.zeros(num_outputs)
        self.left_weighted_num_samples = np.zeros(num_outputs)
        self.right_weighted_num_samples = np.zeros(num_outputs)

        # weighted histogram in the node
        self.node_weighted_histogram = np.zeros((num_outputs, max_num_classes), dtype=np.double)
        # weighted histogram in left node with values smaller than threshold
        self.left_weighted_histogram = np.zeros((num_outputs, max_num_classes), dtype=np.double)
        # weighted histogram in right node with values bigger than threshold
        self.right_weighted_histogram = np.zeros((num_outputs, max_num_classes), dtype=np.double)

        self.threshold_indice = 0

    @property
    def get_node_impurity(self):
        return np.sum(self.node_impurity) / self.num_outputs

    @property
    def get_left_impurity(self):
        return np.sum(self.left_impurity) / self.num_outputs
    
    @property
    def get_right_impurity(self):
        return np.sum(self.right_impurity) / self.num_outputs
    
    # get the weighted histogram for the current node
    @property
    def get_weighted_histogram(self):
        return self.node_weighted_histogram

    def compute_impurity_improvement(self):
        """This method computes the improvement in impurity when a split occurs.
        The weighted impurity improvement equation is the following:

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)
        where N is the total number of samples, N_t is the number of samples
        at the current node, N_t_L is the number of samples in the left child,
        and N_t_R is the number of samples in the right child
        """
        impurity_improvement = np.zeros(self.num_outputs)
        # each output
        for o in range(self.num_outputs):
            impurity_improvement[o] += \
                (self.node_weighted_num_samples[o] / self.num_samples) * (self.node_impurity[o] - \
                    (self.left_weighted_num_samples[o] / self.node_weighted_num_samples[o]) * self.left_impurity[o] - \
                        (self.right_weighted_num_samples[o] / self.node_weighted_num_samples[o]) * self.right_impurity[o])

        return np.sum(impurity_improvement) / self.num_outputs


class Gini(Criterion):
    def __init__(self, 
                 num_outputs, 
                 num_samples, 
                 max_num_classes,
                 num_classes_list,                 
                 class_weights):
        super().__init__(num_outputs, 
                         num_samples, 
                         max_num_classes, 
                         num_classes_list, 
                         class_weights)
        
    
    def _compute_impurity(self, histogram):
        """impurity of a weighted class histogram using the Gini criterion.
        """
        sum_count = 0
        sum_count_squared = 0

        for c in range(len(histogram)):
            sum_count += histogram[c]
            sum_count_squared += histogram[c] * histogram[c]
        
        impurity = (1.0 - sum_count_squared / (sum_count*sum_count)) if (sum_count > 0.0) else 0.0
        return impurity

    def compute_node_histogram(self, y, sample_indices, start, end):
        """compute weighted class histograms for current node.
        """
        # each output
        for o in range(self.num_outputs):
            # Calculate class histogram
            # 1d array to hold the class histogram
            histogram = np.zeros(self.max_num_classes)

            for i in range(start, end):
                histogram[y[sample_indices[i] * self.num_outputs + o]] += 1.0

            weighted_count = 0.0
            self.node_weighted_num_samples[0] = 0.0
            for c in range(self.num_classes_list[o]):
                weighted_count = self.class_weights[o * self.max_num_classes + c] * histogram[c]
                self.node_weighted_histogram[o, c] = weighted_count
                self.node_weighted_num_samples[o] += weighted_count
            
    def compute_node_impurity(self):
        """Evaluate the impurity of the current node.
        Evaluate the Gini criterion as impurity of the current node,
        """
        # each output
        for o in range(self.num_outputs):
            self.node_impurity[o] = self._compute_impurity(self.node_weighted_histogram[o])

    def init_children_histogram(self):
        """Initialize class histograms for all outputs 
        for using a threshold on samples with values,
        """
        # each output
        for o in range(self.num_outputs):
            # init class histogram for left child and right child 
            # value of left child is 0, value of right child is current node value
            
            for c in range(self.num_classes_list[o]):
                self.left_weighted_histogram[o, c] = 0.0
                self.right_weighted_histogram[o, c] = self.node_weighted_histogram[o, c]
            
            self.left_weighted_num_samples[o] = 0.0
            self.right_weighted_num_samples[o] = self.node_weighted_num_samples[o]
        
        self.threshold_indice = 0
    
    def compute_children_impurity(self):
        # each output
        for o in range(self.num_outputs):
            self.left_impurity[o] = self._compute_impurity(self.left_weighted_histogram[o])
            self.right_impurity[o] = self._compute_impurity(self.right_weighted_histogram[o])

    def update_children_histogram(self, y, sample_indices, new_threshold_indice):
        """update class histograms of child nodes with new threshold
        """
        # each output
        for o in range(self.num_outputs):
            histogram = np.zeros(self.max_num_classes)

            for i in range(self.threshold_indice, new_threshold_indice):
                histogram[y[sample_indices[i] * self.num_outputs + o]] += 1.0

            weighted_count = 0.0
            for c in range(self.num_classes_list[o]):
                weighted_count = self.class_weights[o * self.max_num_classes + c] * histogram[c]

                # left child node
                # add class histogram for samples[idx:new_idx]
                self.left_weighted_histogram[o, c] += weighted_count
                self.left_weighted_num_samples[o] += weighted_count

                # right child node
                self.right_weighted_histogram[o, c] -= weighted_count
                self.right_weighted_num_samples[o] -= weighted_count

        self.threshold_indice = new_threshold_indice

    

    



