from DT import DecisionTree  # Assuming DT is the module where DecisionTree class is defined
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import time

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_feature=None):
        # Constructor for the RandomForest class
        self.n_trees = n_trees  # Number of decision trees in the forest
        self.max_depth = max_depth  # Maximum depth for each decision tree
        self.min_samples_split = min_samples_split  # Minimum samples required to split a node
        self.n_features = n_feature  # Number of features to consider for each split
        self.trees = []  # List to store individual decision trees

    def fit(self, X, y):
        # Fit the random forest to the training data
        self.trees = []
        for _ in range(self.n_trees):
            # Create and fit a decision tree with bootstrap samples
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        # Generate bootstrap samples for training a tree
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        # Find the most common label in a set
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        # Make predictions using the ensemble of decision trees
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions

def evaluate_classifier(classifier, X_train, X_test, Y_train, Y_test):
    # Evaluate the classifier using various metrics
    start_time = time.time()

    # Fit the model on the training data
    classifier.fit(X_train, Y_train.flatten())

    # Make predictions on the test data
    Y_pred = classifier.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, average='weighted')
    recall = recall_score(Y_test, Y_pred, average='weighted')
    f1 = f1_score(Y_test, Y_pred, average='weighted')

    # Calculate confusion matrix
    confusion_mat = confusion_matrix(Y_test, Y_pred)

    # Calculate class-specific metrics
    class_metrics = {}
    specificity_scores = {}
    unique_labels = np.unique(Y_test)
    for label in unique_labels:
        label_mask = Y_test.flatten() == label
        class_accuracy = accuracy_score(Y_test[label_mask], Y_pred[label_mask])
        class_precision = precision_score(Y_test[label_mask], Y_pred[label_mask], average='weighted')
        class_recall = recall_score(Y_test[label_mask], Y_pred[label_mask], average='weighted')
        class_f1 = f1_score(Y_test[label_mask], Y_pred[label_mask], average='weighted')
        # Calculate specificity
        tn, fp, fn, tp = confusion_matrix(Y_test == label, Y_pred == label).ravel()
        specificity = tn / (tn + fp)
        specificity_scores[f"Class {label}"] = specificity

        class_metrics[f"Class {label}"] = {
            'Accuracy': class_accuracy,
            'Precision': class_precision,
            'Recall': class_recall,
            'F1 Score': class_f1,
            'Specificity': specificity
        }
    global_specificity = np.mean(list(specificity_scores.values()))

    # Calculate execution time
    execution_time = time.time() - start_time

    # Print evaluation metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"\nGlobal Specificity: {global_specificity:.4f}")

    print(confusion_mat)

    print("\nClass-specific Metrics:")
    for class_name, metrics in class_metrics.items():
        print(f"\n{class_name}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

    print(f"\nExecution Time: {execution_time:.4f} seconds")

def split_data_by_class(X, y, test_size=0.2, random_state=None):
    # Split data into training and testing sets for each class
    unique_classes = np.unique(y)
    X_train, X_test, Y_train, Y_test = [], [], [], []

    for class_label in unique_classes:
        # Filter examples for the specific class
        X_class = X[y.flatten() == class_label]
        y_class = y[y.flatten() == class_label]

        # Split data for this class
        X_train_class, X_test_class, Y_train_class, Y_test_class = train_test_split(
            X_class, y_class, test_size=test_size, random_state=random_state
        )

        # Add data for this class to the global list
        X_train.append(X_train_class)
        X_test.append(X_test_class)
        Y_train.append(Y_train_class)
        Y_test.append(Y_test_class)

    # Concatenate lists to get final training and testing data
    X_train = np.concatenate(X_train)
    X_test = np.concatenate(X_test)
    Y_train = np.concatenate(Y_train)
    Y_test = np.concatenate(Y_test)

    return X_train, X_test, Y_train, Y_test
