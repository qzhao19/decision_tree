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

        # weighted number of samples in the node
        self.weighted_num_samples_node = np.zeros(num_outputs)

        # weighted histogram in the node
        self.weighted_histogram_node = np.zeros((num_outputs, num_classes_max))



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
                weighted_count = self.class_weights[o * self.num_outputs + c] * histogram[c]
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
        """Compute the impurity of the current node
        """
        # each output
        for o in range(self.num_outputs):
            self.impurity_node[o] = self._compute_impurity(self.weighted_histogram_node[o])






































