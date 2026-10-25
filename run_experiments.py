"""Reproduce Lab 3 experiments and write measured results to JSON."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from grader_contracts.linear_models import ClassificationInput, RegressionInput
from linear_models import train_linear_regression, train_logistic_classifier, compare_classical_models, _get_metrics

ROOT = Path(__file__).resolve().parent
SEED = 42


def main():
    students = pd.read_csv(ROOT / 'Student_Performance.csv')
    students['Extracurricular Activities'] = students['Extracurricular Activities'].map({'No': 0, 'Yes': 1})
    x = students.drop(columns=['Performance Index']).to_numpy(dtype=float)
    y = students['Performance Index'].to_numpy(dtype=float)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED)
    regression = train_linear_regression(RegressionInput(x_train, y_train, x_test, y_test))
    mse = float(mean_squared_error(y_test, regression.predictions))

    german = pd.read_csv(ROOT / 'german.csv', sep=';')
    features = german.drop(columns=['Creditability']).to_numpy(dtype=float)
    labels = german['Creditability'].to_numpy(dtype=int)
    cx_train, cx_test, cy_train, cy_test = train_test_split(
        features, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    data = ClassificationInput(cx_train, cy_train, cx_test, cy_test)
    own = train_logistic_classifier(data)
    own_metrics = _get_metrics(cy_test, own.predictions, own.probabilities)
    other = compare_classical_models(data)
    # Optional reference baseline: sklearn is used only outside the own-GD
    # implementation and outside the required three-model comparison.
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(cx_train)
    test_scaled = scaler.transform(cx_test)
    sklearn_logistic = LogisticRegression(max_iter=1000, random_state=SEED)
    sklearn_logistic.fit(train_scaled, cy_train)
    reference_probabilities = sklearn_logistic.predict_proba(test_scaled)[:, 1]
    reference_predictions = sklearn_logistic.predict(test_scaled)
    reference_metrics = _get_metrics(cy_test, reference_predictions, reference_probabilities)
    assert abs(own_metrics.roc_auc - other.logistic_regression.roc_auc) < 1e-12
    output = {
        'linear_regression': {'test_mse': mse},
        'classification': {
            'own_logistic': asdict(own_metrics),
            'logistic_regression': asdict(reference_metrics),
            'decision_tree': asdict(other.decision_tree),
            'knn': asdict(other.knn),
        },
    }
    (ROOT / 'experiment_results.json').write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(output, indent=2))
    print('Regression initial/final train MSE:', regression.loss_history[0], regression.loss_history[-1])
    print('Classification initial/final train BCE:', own.loss_history[0], own.loss_history[-1])
    print('Data shapes:', x_train.shape, x_test.shape, cx_train.shape, cx_test.shape)


if __name__ == '__main__':
    main()
