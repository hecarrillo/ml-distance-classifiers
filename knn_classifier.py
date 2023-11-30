import numpy as np
from scipy.spatial.distance import cdist
from collections import Counter

def euclidean_distance(a, b):
    """
    Computes the Euclidean distance between two vectors.
    """
    return np.linalg.norm(np.array(a) - np.array(b))

def manhattan_distance(a, b):
    """
    Computes the Manhattan distance between two vectors.
    """
    return np.sum(np.abs(np.array(a) - np.array(b)))

class KNNClassifier:
    def __init__(self, k=3, distance_metric='euclidean'):
        """
        Initializes the KNN classifier with the number of neighbors and distance metric.
        """
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = []
        self.y_train = []

    def fit(self, X, y):
        """
        Fits the classifier with the training data.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predicts the class labels for the provided data.
        """
        predictions = []
        for x in X:
            # Compute distances from the test point to all training points
            distances = [euclidean_distance(x, x_train) if self.distance_metric == 'euclidean' 
                         else manhattan_distance(x, x_train) for x_train in self.X_train]

            # Sort by distance and get the indices of the nearest neighbors
            k_indices = np.argsort(distances)[:self.k]

            # Extract the labels of the nearest neighbors
            k_nearest_labels = [self.y_train[i] for i in k_indices]

            # Determine the most common class label among the nearest neighbors
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
        return predictions
