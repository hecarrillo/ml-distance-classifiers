import input_preprocess
from knn_classifier import KNNClassifier
from min_distance_classifier import MinimumDistanceClassifier

# File and delimiter
FILE_NAME = 'maternal.csv'
DELIMITATOR = ','

# Read and preprocess the data
document = input_preprocess.read_from_file(FILE_NAME, DELIMITATOR)
for content in document:
    content[-1] = content[-1].strip()
    content[0] = content[0].replace("ï»¿", "")

attributes_number = len(document[0])
patterns_number = len(document)
attributes = input_preprocess.label_attributes(document, attributes_number, patterns_number)
attributes_input, attributes_output = input_preprocess.default_selection(attributes)

# Print attribute data (optional, for verification)
input_preprocess.print_attr_data(attributes, attributes_input, attributes_output, attributes_number, patterns_number)

# Select a subset of data for training and testing
new_matrix = input_preprocess.select_subset(document, attributes_number, patterns_number, DELIMITATOR)

# Convert the new_matrix to a suitable format for training and testing
# This needs to be adapted based on how your data looks like
X_train = [data[0] for data in new_matrix]  # Inputs
X_train = [[float(value) for value in row] for row in X_train]
y_train = [data[1] for data in new_matrix]  # Outputs
y_train = [row[0] for row in y_train]


print("x train", X_train[1:15])
print("y train", y_train[1:15])
# Initialize and train KNN Classifier
knn_classifier = KNNClassifier(k=3, distance_metric='euclidean')
knn_classifier.fit(X_train, y_train)

# Example prediction with KNN (replace X_test with actual test data)
# X_test = [[...], [...]]  # Test data
# predictions_knn = knn_classifier.predict(X_test)

# Initialize and train Minimum Distance Classifier
min_dist_classifier = MinimumDistanceClassifier(distance_metric='euclidean')
min_dist_classifier.fit(X_train, y_train)

# Example prediction with Minimum Distance Classifier (replace X_test with actual test data)
# predictions_min_dist = min_dist_classifier.predict(X_test)

# Print predictions (for both classifiers)
# print("KNN Predictions:", predictions_knn)
# print("Minimum Distance Predictions:", predictions_min_dist)
