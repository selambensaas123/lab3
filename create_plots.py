
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from grader_contracts.linear_models import RegressionInput, ClassificationInput
from linear_models import train_linear_regression, train_logistic_classifier


# Find the folder containing this Python file
ROOT = Path(__file__).resolve().parent

# Use the same random seed as the experiments
SEED = 42


# ----- Linear regression -----

students = pd.read_csv(ROOT / "Student_Performance.csv")

# Convert Yes/No values into numbers
students["Extracurricular Activities"] = students[
    "Extracurricular Activities"
].map({"No": 0, "Yes": 1})

X = students.drop(columns=["Performance Index"]).to_numpy(dtype=float)
y = students["Performance Index"].to_numpy(dtype=float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

regression = train_linear_regression(
    RegressionInput(X_train, y_train, X_test, y_test)
)

# Plot the MSE history
plt.figure(figsize=(8, 5))
plt.plot(regression.loss_history)
plt.title("Linear Regression - Training MSE")
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error (MSE)")
plt.grid(True)
plt.tight_layout()
plt.savefig(ROOT / "loss_linear_regression.png", dpi=150)
plt.close()

print("Created: loss_linear_regression.png")


# ----- Logistic regression -----

german = pd.read_csv(ROOT / "german.csv", sep=";")

X = german.drop(columns=["Creditability"]).to_numpy(dtype=float)
y = german["Creditability"].to_numpy(dtype=int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

classification = train_logistic_classifier(
    ClassificationInput(X_train, y_train, X_test, y_test)
)

# Plot the binary cross-entropy history
plt.figure(figsize=(8, 5))
plt.plot(classification.loss_history)
plt.title("Logistic Regression - Training BCE")
plt.xlabel("Iteration")
plt.ylabel("Binary Cross-Entropy (BCE)")
plt.grid(True)
plt.tight_layout()
plt.savefig(ROOT / "loss_logistic_regression.png", dpi=150)
plt.close()

print("Created: loss_logistic_regression.png")
