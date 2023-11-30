import numpy as np

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

class MinimumDistanceClassifier:
    def __init__(self, distance_metric='euclidean'):
        """
        Initializes the Minimum Distance classifier with the specified distance metric.
        """
        self.distance_metric = distance_metric
        self.means = {}
        self.classes = []

    def fit(self, X, y):
        """
        Fits the classifier with the training data.
        """
        self.classes = np.unique(y)
        for cls in self.classes:
            self.means[cls] = np.mean([X[i] for i in range(len(X)) if y[i] == cls], axis=0)

    def predict(self, X):
        """
        Predicts the class labels for the provided data.
        """
        predictions = []
        for x in X:
            # Compute distances from the test point to the mean of each class
            distances = {cls: euclidean_distance(x, mean) if self.distance_metric == 'euclidean' 
                         else manhattan_distance(x, mean) for cls, mean in self.means.items()}

            # Choose the class with the minimum distance
            predictions.append(min(distances, key=distances.get))
        return predictions
