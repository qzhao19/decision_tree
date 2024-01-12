import numpy as np

class Criterion(object):
    def __init__(self, class_weight, 
                 num_outputs, 
                 num_samples, 
                 num_classes_max, 
                 num_classes_list):
        self.class_weight = class_weight
        self.num_outputs = num_outputs
        self.num_samples = num_samples
        self.num_classes_max = num_classes_max
        self.num_classes_list = num_classes_list
        


class Gini(Criterion):
    def __init__(self, class_weight, 
                 num_outputs, 
                 num_samples, 
                 num_classes_max,
                 num_classes_list):
        super().__init__(class_weight, 
                         num_outputs, 
                         num_samples, 
                         num_classes_max, 
                         num_classes_list)
        
        # impurity of in the current node
        self.impurity_node = np.zeros(num_outputs)
        # impurity of in the left node with values smaller than threshold
        self.impurity_left = np.zeros(num_outputs)
        # impurity of in the right node with values bigger that threshold
        self.impurity_right = np.zeros(num_outputs)

        # weighted number of samples in the node, left child and right child
        self.weighted_num_samples_node = np.zeros(num_outputs)
        self.weighted_num_samples_left = np.zeros(num_outputs)
        self.weighted_num_samples_right = np.zeros(num_outputs)


        # weighted histogram in the node
        self.weighted_histogram_node = np.zeros((num_outputs, num_classes_max))
        # weighted histogram in left node with values smaller than threshold
        self.weighted_histogram_left = np.zeros((num_outputs, num_classes_max))
        # weighted histogram in right node with values bigger than threshold
        self.weighted_histogram_right = np.zeros((num_outputs, num_classes_max))

        self.threshold_indice = 0

    def compute_node_histogram(self, y, sample_indices, start, end):
        """compute weighted class histograms for current node.
        """
        # each output
        for o in range(self.num_outputs):

            # Calculate class histogram
            # 1d array to hold the class histogram
            histogram = np.zeros(self.num_classes_max)

            for i in range(start, end):
                histogram[y[sample_indices[i] * self.num_outputs + o]] += 1

            weighted_count = 0
            for c in range(self.num_classes_list[o]):
                weighted_count = self.class_weights[o * self.num_classes_max + c] * histogram[c]
                self.weighted_histogram_node[o, c] = weighted_count
                self.weighted_num_samples_node[o] += weighted_count
            

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


    def compute_node_impurity(self):
        """Evaluate the impurity of the current node.
        Evaluate the Gini criterion as impurity of the current node,
        """
        # each output
        for o in range(self.num_outputs):
            self.impurity_node[o] = self._compute_impurity(self.weighted_histogram_node[o])


    def compute_threshold_impurity(self):
        # each output
        for o in range(self.num_outputs):
            self.impurity_left[o] = self._compute_impurity(self.weighted_histogram_left[o])
            self.impurity_right[o] = self._compute_impurity(self.weighted_histogram_right[o])


    def init_threshold_histogram(self):
        """Initialize class histograms for all outputs 
        for using a threshold on samples with values,
        """
        # each output
        for o in range(self.num_outputs):
            # init class histogram for left child and right child 
            # value of left child is 0, value of right child is current node value
            
            for c in range(self.num_classes_list[o]):
                self.weighted_histogram_left[o, c] = 0.0
                self.weighted_histogram_right[c, o] = self.weighted_histogram_node[o, c]
            
            self.weighted_num_samples_left[o] = 0
            self.weighted_num_samples_right[o] = self.weighted_num_samples_node[o]
        
        self.threshold_indice = 0


    def update_threshold_histogram(self, y, sample_indices, new_indice):
        """
        """
        # each output
        for o in range(self.num_outputs):
            histogram = np.zeros(self.num_classes_max)

            for i in range(self.threshold_indice, new_indice):
                histogram[y[sample_indices[i] * self.num_outputs + o]] += 1

            weighted_count = 0
            for c in range(self.num_classes_list[o]):
                weighted_count = self.class_weights[o * self.num_classes_max + c] * histogram[c]

                # left child node
                # add class histogram for samples[idx:new_idx]
                self.weighted_histogram_left[o, c] += weighted_count
                self.weighted_num_samples_left[o] += weighted_count

                # right child node
                self.weighted_histogram_right[o, c] -= weighted_count
                self.weighted_num_samples_right[o] -= weighted_count

        self.threshold_indice = new_indice

    
    def compute_impurity_improvement(self):
        """This method computes the improvement in impurity when a split occurs.
        The weighted impurity improvement equation is the following:

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        """
        impurity_improvement = np.zeros(self.num_outputs)
        # each output
        for o in range(self.num_outputs):
            impurity_improvement[o] += (self.weighted_num_samples_node[o] / self.num_samples)



























