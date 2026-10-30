
"""
neuro_fuzzy.py

Project: Intelligent Healthcare Support System
Technique: Neuro-Fuzzy System (simplified ANFIS-style model)

Purpose:
    Combine fuzzy membership functions with trainable parameters
    to classify heart disease cases.

How it works:
    1. Convert age, blood pressure, and cholesterol into fuzzy memberships.
    2. Combine memberships to create fuzzy-rule activations.
    3. Normalize the rule activations.
    4. Learn output weights from the training data.
    5. Evaluate the model on validation and test data.

Important:
    - Fuzzy membership functions are implemented manually.
    - This is a simplified educational neuro-fuzzy model.
    - It is not a complete implementation of canonical ANFIS.
    - This model is not a medical diagnostic tool.
"""

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data_preprocessing import load_and_preprocess


# ---------------------------------------------------------
# 1. MANUAL FUZZY MEMBERSHIP FUNCTIONS
# ---------------------------------------------------------

def triangular_membership(x, a, b, c):
    """Return membership in a triangular fuzzy set."""
    if x < a or x > c:
        return 0.0

    if x == b:
        return 1.0

    if x < b:
        if b == a:
            return 1.0
        return (x - a) / (b - a)

    if c == b:
        return 1.0

    return (c - x) / (c - b)


def trapezoidal_membership(x, a, b, c, d):
    """Return membership in a trapezoidal fuzzy set."""
    if x < a or x > d:
        return 0.0

    if b <= x <= c:
        return 1.0

    if x < b:
        if b == a:
            return 1.0
        return (x - a) / (b - a)

    if x > c:
        if d == c:
            return 1.0
        return (d - x) / (d - c)

    return 0.0


# ---------------------------------------------------------
# 2. FUZZIFY CLINICAL INPUTS
# ---------------------------------------------------------

def fuzzify_age(age):
    """Represent age using Young, Middle, and Older fuzzy sets."""
    return [
        trapezoidal_membership(age, 20, 20, 35, 50),
        triangular_membership(age, 35, 52, 70),
        trapezoidal_membership(age, 55, 70, 100, 100),
    ]


def fuzzify_blood_pressure(bp):
    """Represent blood pressure using Low, Normal, and High sets."""
    return [
        trapezoidal_membership(bp, 0, 0, 90, 110),
        triangular_membership(bp, 90, 120, 150),
        trapezoidal_membership(bp, 130, 150, 220, 220),
    ]


def fuzzify_cholesterol(chol):
    """Represent cholesterol using Normal, Borderline, and High sets."""
    return [
        trapezoidal_membership(chol, 0, 0, 180, 220),
        triangular_membership(chol, 180, 225, 270),
        trapezoidal_membership(chol, 240, 270, 600, 600),
    ]


# ---------------------------------------------------------
# 3. EXTRACT THE THREE REQUIRED FEATURES
# ---------------------------------------------------------

def extract_clinical_features(X, feature_names):
    """
    Find age, trestbps, and chol in the preprocessed feature matrix.

    The preprocessing module may one-hot encode categorical features.
    We therefore locate the original numeric feature columns by name.
    """
    required_features = ["age", "trestbps", "chol"]

    indices = {}

    for feature in required_features:
        matching_indices = [
            i for i, name in enumerate(feature_names)
            if name == feature
        ]

        if not matching_indices:
            raise ValueError(
                f"Could not find required feature '{feature}' "
                "in the preprocessing output."
            )

        indices[feature] = matching_indices[0]

    age_values = X[:, indices["age"]]
    bp_values = X[:, indices["trestbps"]]
    chol_values = X[:, indices["chol"]]

    return age_values, bp_values, chol_values


# ---------------------------------------------------------
# 4. GENERATE FUZZY RULE ACTIVATIONS
# ---------------------------------------------------------

def create_fuzzy_features(X, feature_names):
    """
    Generate nine fuzzy rule activation features.

    Each input has three membership values:
        Age: Young, Middle, Older
        BP: Low, Normal, High
        Cholesterol: Normal, Borderline, High

    This simplified model uses nine selected rules rather than
    every possible combination of membership functions.
    """

    age_values, bp_values, chol_values = extract_clinical_features(
        X,
        feature_names
    )

    fuzzy_rows = []

    for age, bp, chol in zip(
        age_values,
        bp_values,
        chol_values
    ):
        age_sets = fuzzify_age(age)
        bp_sets = fuzzify_blood_pressure(bp)
        chol_sets = fuzzify_cholesterol(chol)

        young, middle, older = age_sets
        low_bp, normal_bp, high_bp = bp_sets
        normal_chol, borderline_chol, high_chol = chol_sets

        # Each rule uses MIN as the fuzzy AND operator.
        rule_activations = [
            min(older, high_chol),
            min(high_bp, high_chol),
            min(older, high_bp),
            min(middle, borderline_chol),
            min(young, normal_bp, normal_chol),
            min(normal_bp, normal_chol),
            older,
            high_bp,
            high_chol,
        ]

        fuzzy_rows.append(rule_activations)

    return np.asarray(fuzzy_rows, dtype=float)


# ---------------------------------------------------------
# 5. NORMALIZE FUZZY RULE ACTIVATIONS
# ---------------------------------------------------------

def normalize_rule_activations(fuzzy_features):
    """
    Normalize each row so its rule activations sum to 1.

    Normalization helps the model combine the relative strengths
    of the activated rules.

    If no rule activates, all normalized activations remain zero.
    """
    row_sums = np.sum(
        fuzzy_features,
        axis=1,
        keepdims=True
    )

    # Avoid division by zero.
    row_sums[row_sums == 0] = 1.0

    return fuzzy_features / row_sums


# ---------------------------------------------------------
# 6. TRAIN THE NEURO-FUZZY OUTPUT LAYER
# ---------------------------------------------------------

def train_neuro_fuzzy_model(X_train, y_train, feature_names):
    """
    Create fuzzy features and learn their output weights.

    Logistic regression learns how strongly each fuzzy rule
    contributes to the binary classification.

    The membership functions and fuzzy rules are fixed in this
    simplified version; they are not learned from the data.
    """
    fuzzy_train = create_fuzzy_features(
        X_train,
        feature_names
    )

    normalized_train = normalize_rule_activations(
        fuzzy_train
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(
        normalized_train,
        np.asarray(y_train).ravel()
    )

    return model


# ---------------------------------------------------------
# 7. EVALUATE THE MODEL
# ---------------------------------------------------------

def evaluate_neuro_fuzzy_model(
    model,
    X,
    y,
    feature_names,
    dataset_name
):
    """Evaluate classification performance on a supplied dataset."""

    fuzzy_features = create_fuzzy_features(
        X,
        feature_names
    )

    normalized_features = normalize_rule_activations(
        fuzzy_features
    )

    probabilities = model.predict_proba(
        normalized_features
    )[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

    y = np.asarray(y).astype(int).ravel()

    accuracy = accuracy_score(y, predictions)
    f1 = f1_score(y, predictions, zero_division=0)

    if len(np.unique(y)) == 2:
        auc = roc_auc_score(y, probabilities)
    else:
        auc = float("nan")

    print(f"\n===== {dataset_name} Results =====")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")

    return {
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": auc,
    }


# ---------------------------------------------------------
# 8. MAIN PROGRAM
# ---------------------------------------------------------

def main():
    """Load the data, train the model, and evaluate its performance."""

    print("Loading preprocessed healthcare data...")

    data = load_and_preprocess()
    print("\nAvailable top-level keys:", data.keys())
    print("Fuzzy data keys:", data["fuzzy"].keys())

    # Use the unscaled, one-hot encoded feature matrices so that
    # age, blood pressure, and cholesterol retain their real units.
    X_train = np.asarray(data["fuzzy"]["X_train"], dtype=float)
    X_val = np.asarray(data["fuzzy"]["X_val"], dtype=float)
    X_test = np.asarray(data["fuzzy"]["X_test"], dtype=float)

    y_train = np.asarray(data["y_train"], dtype=int)
    y_val = np.asarray(data["y_val"], dtype=int)
    y_test = np.asarray(data["y_test"], dtype=int)

    # The preprocessing module should provide the feature names
    # in the same order as the fuzzy feature matrix.
# Get the feature names from the preprocessing output.
    feature_names = data["feature_columns"]

    print(f"Training samples: {len(y_train)}")
    print(f"Validation samples: {len(y_val)}")
    print(f"Test samples: {len(y_test)}")

    print("\nTraining Neuro-Fuzzy model...")

    model = train_neuro_fuzzy_model(
        X_train,
        y_train,
        feature_names
    )

    evaluate_neuro_fuzzy_model(
        model,
        X_val,
        y_val,
        feature_names,
        "Validation"
    )

    evaluate_neuro_fuzzy_model(
        model,
        X_test,
        y_test,
        feature_names,
        "Test"
    )

    print("\nNeuro-Fuzzy training and evaluation completed.")
    print("This academic model is not a medical diagnostic tool.")


if __name__ == "__main__":
    main()
