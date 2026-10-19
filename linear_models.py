"""Linear and logistic models trained with batch gradient descent."""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    # Normalize using training data only.
    x_train = np.asarray(data.train_features, dtype=float)
    x_test = np.asarray(data.test_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).reshape(-1)
    means = x_train.mean(axis=0)
    scales = x_train.std(axis=0)
    scales[scales == 0] = 1.0
    train = (x_train - means) / scales
    test = (x_test - means) / scales

    weights = np.zeros(train.shape[1])
    bias = float(y_train.mean())
    history = []
    number_of_rows = len(y_train)
    for epoch in range(data.epochs):
        predictions = train @ weights + bias
        errors = predictions - y_train
        gradient_w = (2.0 / number_of_rows) * (train.T @ errors)
        gradient_b = float(2.0 * errors.mean())
        weights = weights - data.learning_rate * gradient_w
        bias = bias - data.learning_rate * gradient_b
        new_errors = train @ weights + bias - y_train
        history.append(float(np.mean(new_errors ** 2)))

    # Convert weights back to the original feature scale.
    original_weights = weights / scales
    original_bias = float(bias - means @ original_weights)
    test_predictions = x_test @ original_weights + original_bias
    return RegressionResult(original_weights, original_bias, history, test_predictions)


def _sigmoid(values):
    # Clipping prevents overflow in exp for large values.
    return 1.0 / (1.0 + np.exp(-np.clip(values, -500, 500)))


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    x_train = np.asarray(data.train_features, dtype=float)
    x_test = np.asarray(data.test_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).reshape(-1)
    means = x_train.mean(axis=0)
    scales = x_train.std(axis=0)
    scales[scales == 0] = 1.0
    train = (x_train - means) / scales
    test = (x_test - means) / scales
    weights = np.zeros(train.shape[1])
    fraction = float(np.clip(y_train.mean(), 1e-6, 1 - 1e-6))
    bias = float(np.log(fraction / (1 - fraction)))
    history = []

    for epoch in range(data.epochs):
        probabilities = _sigmoid(train @ weights + bias)
        errors = probabilities - y_train
        gradient_w = (train.T @ errors) / len(y_train)
        gradient_b = float(errors.mean())
        weights = weights - data.learning_rate * gradient_w
        bias = bias - data.learning_rate * gradient_b
        new_probabilities = _sigmoid(train @ weights + bias)
        clipped = np.clip(new_probabilities, 1e-12, 1 - 1e-12)
        loss = -np.mean(y_train * np.log(clipped) + (1 - y_train) * np.log(1 - clipped))
        history.append(float(loss))

    original_weights = weights / scales
    original_bias = float(bias - means @ original_weights)
    test_probabilities = _sigmoid(test @ weights + bias)
    test_predictions = (test_probabilities >= 0.5).astype(int)
    return ClassificationResult(original_weights, original_bias, history, test_probabilities, test_predictions)


def _get_metrics(y_true, predictions, probabilities):
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        accuracy=float(accuracy_score(y_true, predictions)),
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    # All models use the same training and testing rows.
    x_train = np.asarray(data.train_features, dtype=float)
    x_test = np.asarray(data.test_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=int).reshape(-1)
    y_test = np.asarray(data.test_target, dtype=int).reshape(-1)
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(x_train)
    scaled_test = scaler.transform(x_test)

    logistic = LogisticRegression(max_iter=1000, random_state=42)
    logistic.fit(scaled_train, y_train)
    logistic_probs = logistic.predict_proba(scaled_test)[:, 1]
    logistic_metrics = _get_metrics(y_test, logistic.predict(scaled_test), logistic_probs)

    tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree.fit(x_train, y_train)
    tree_probs = tree.predict_proba(x_test)[:, 1]
    tree_metrics = _get_metrics(y_test, tree.predict(x_test), tree_probs)

    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(scaled_train, y_train)
    knn_probs = knn.predict_proba(scaled_test)[:, 1]
    knn_metrics = _get_metrics(y_test, knn.predict(scaled_test), knn_probs)

    return BaselineComparisonResult(logistic_metrics, tree_metrics, knn_metrics)
