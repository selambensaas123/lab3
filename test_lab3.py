"""Small checks for the Lab 3 implementations."""

import numpy as np
from grader_contracts.linear_models import RegressionInput, ClassificationInput
from linear_models import train_linear_regression, train_logistic_classifier, compare_classical_models


def test_regression_learns_simple_line():
    x = np.arange(20, dtype=float).reshape(-1, 1)
    y = 3 * x[:, 0] + 2
    result = train_linear_regression(RegressionInput(x, y, x, y))
    assert np.mean((result.predictions - y) ** 2) < 0.01
    assert len(result.loss_history) == 800
    assert result.loss_history[-1] < result.loss_history[0]


def test_logistic_shapes_and_loss():
    x = np.array([[-3], [-2], [-1], [1], [2], [3]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])
    result = train_logistic_classifier(ClassificationInput(x, y, x, y))
    assert len(result.probabilities) == 6
    assert np.array_equal(result.predictions, y)
    assert result.loss_history[-1] < result.loss_history[0]


def test_baseline_metrics_are_valid():
    x = np.arange(60, dtype=float).reshape(-1, 1)
    y = (x[:, 0] > 29).astype(int)
    result = compare_classical_models(ClassificationInput(x, y, x, y))
    for metrics in [result.logistic_regression, result.decision_tree, result.knn]:
        assert 0 <= metrics.roc_auc <= 1
        assert 0 <= metrics.f1 <= 1
