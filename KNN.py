import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.Y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # compute the distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]

def evaluate_classifier(classifier, X_train, X_test, Y_train, Y_test):
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
    unique_classes = np.unique(y)
    X_train, X_test, Y_train, Y_test = [], [], [], []

    for class_label in unique_classes:
        # Filtrer les exemples pour la classe spécifique
        X_class = X[y.flatten() == class_label]
        y_class = y[y.flatten() == class_label]

        # Diviser les données pour cette classe
        X_train_class, X_test_class, Y_train_class, Y_test_class = train_test_split(
            X_class, y_class, test_size=test_size, random_state=random_state
        )

        # Ajouter les données de cette classe à la liste globale
        X_train.append(X_train_class)
        X_test.append(X_test_class)
        Y_train.append(Y_train_class)
        Y_test.append(Y_test_class)

    # Concaténer les listes pour obtenir les données d'apprentissage et de test finales
    X_train = np.concatenate(X_train)
    X_test = np.concatenate(X_test)
    Y_train = np.concatenate(Y_train)
    Y_test = np.concatenate(Y_test)

    return X_train, X_test, Y_train, Y_test


