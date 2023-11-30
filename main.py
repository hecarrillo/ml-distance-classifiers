import input_preprocess
import json
from knn_classifier import KNNClassifier
from min_distance_classifier import MinimumDistanceClassifier

FILE_NAME = str(input("Nombre del archivo: ")) # Ejemplo: "maternal.csv"
DELIMITATOR = str(input("Delimitador: ")) # Ejemplo: ","

# Process the file
document = input_preprocess.read_from_file(FILE_NAME, DELIMITATOR)
# ... (rest of the processing using input_preprocess functions)

# Create and train classifiers
# Example: classifier = KNNClassifier(k=3, distance_metric='euclidean')
#          classifier.fit(X_train, y_train)
#          predictions = classifier.predict(X_test)
# You need to replace X_train, y_train, X_test with actual data

# Similarly for Minimum Distance Classifier
# Example: classifier = MinimumDistanceClassifier(distance_metric='euclidean')
#          classifier.fit(X_train, y_train)
#          predictions = classifier.predict(X_test)
