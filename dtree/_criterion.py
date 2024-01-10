import numpy as np

class Criterion(object):
    def __init__(self, num_classes, num_samples, class_weight):
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.class_weight = class_weight



class Gini(Criterion):
    def __init__(self, num_classes, num_samples, class_weight):
        super().__init__(num_classes, num_samples, class_weight)

        # weighted number of samples in the node
        self.weighted_num_samples_node = 0.0
        # weighted number of samples in the left node with values smaller than threshold
        self.weighted_num_samples_left = 0.0
        # weighted number of samples in the right node with values bigger than threshold
        self.weighted_num_samples_right = 0.0
        
        # impurity of in the node
        self.node_impurity = 0.0

        # weighted histogram in the node
        self.weighted_histogram_node = np.zeros(self.num_classes)
        # weighted histogram in the left node with values smaller than threshold (assigned to left child)
        self.weighted_histogram_left = np.zeros(self.num_classes)
        # weighted histogram in the right node with values bigger than threshold (assigned to right child)
        self.weighted_histogram_right = np.zeros(self.num_classes)


    def compute_node_histogram(self, y, samples, start, end):
        """compute weighted class histograms for current node.
        """
        histogram = np.zeros(self.num_classes)

        # Calculate class histogram
        # 1d array to hold the class histogram
        for i in range(start, end):
            histogram[y[samples[i]]] += 1

        for c in range(self.num_classes):
            weighted_count = self.class_weight[c] * histogram[c]
            self.weighted_histogram_node[c] = weighted_count
            self.weighted_num_samples_node += weighted_count
        
    
    def _compute_impurity(self, histogram):
        """impurity of a weighted class histogram
        """
        sum_count = 0
        sum_count_squared = 0

        for c in histogram:
            sum_count += histogram[c]
            sum_count_squared += histogram[c] * histogram[c]
        
        impurity = (1.0 - sum_count_squared / (sum_count*sum_count)) if (sum_count > 0.0) else 0.0
        return impurity


    def compute_node_impurity(self):
        """
        Compute the impurity of the current node.
        """
        self.node_impurity = self._compute_impurity(self.weighted_histogram_node)

    
    def init_threshold_histogram(self):
        """Initialize class histograms for all outputs for using a threshold on samples with values,
        in the case that all samples have values.
        Assuming: calculate_node_histogram()
        """

        for c in range(self.num_classes):
            self.weighted_histogram_left[c] = 0.0
            self.weighted_histogram_right[c] = self.weighted_histogram_node[c]


        self.weighted_num_samples_left = 0.0
        self.weighted_num_samples_right = self.weighted_num_samples_node
    
        self.node_position_threshold = 0


    def update_threshold_histograms(self, y, samples, new_position):
        histogram = np.zeros(self.num_classes)

        for i in range(self.node_position_threshold, new_position):
            histogram[y[samples[i]]] += 1
        
        for c in range(self.num_classes):
            weighted_count = self.class_weight[c] * histogram[c]

            # left child
            self.weighted_histogram_left[c] += weighted_count
            self.weighted_num_samples_left += weighted_count

            # right child
            self.weighted_histogram_right[c] -= weighted_count
            self.weighted_num_samples_right -= weighted_count

        self.node_position_threshold = new_position