
"""
fuzzy_logic.py

Project: Intelligent Healthcare Support System
Technique: Mamdani Fuzzy Logic

Purpose:
    Estimate an illustrative risk score using:
        1. Age
        2. Blood pressure
        3. Cholesterol

Important:
    - Fuzzy membership functions are implemented manually.
    - Mamdani rules use MIN for AND and MAX for aggregation.
    - Centroid defuzzification converts fuzzy output into a number.
    - This is an educational demonstration, NOT a medical diagnosis.
"""

import numpy as np


# ---------------------------------------------------------
# 1. MEMBERSHIP FUNCTIONS
# ---------------------------------------------------------

def triangular_membership(x, a, b, c):
    """
    Calculate membership in a triangular fuzzy set.

    a = left endpoint
    b = point where membership reaches 1
    c = right endpoint

    Membership is between 0 and 1.
    """
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
    """
    Calculate membership in a trapezoidal fuzzy set.

    a, d = outer endpoints
    b, c = interval where membership is 1
    """
    if x < a or x > d:
        return 0.0

    if b <= x <= c:
        return 1.0

    if a <= x < b:
        if b == a:
            return 1.0
        return (x - a) / (b - a)

    if c < x <= d:
        if d == c:
            return 1.0
        return (d - x) / (d - c)

    return 0.0


# ---------------------------------------------------------
# 2. FUZZIFICATION
# Convert crisp inputs into fuzzy membership values.
# ---------------------------------------------------------

def fuzzify_age(age):
    """Find the membership of age in Young, Middle, and Older sets."""
    return {
        "young": trapezoidal_membership(age, 20, 20, 35, 50),
        "middle": triangular_membership(age, 35, 52, 70),
        "older": trapezoidal_membership(age, 55, 70, 100, 100),
    }


def fuzzify_blood_pressure(bp):
    """Find the membership of blood pressure in Low, Normal, and High sets."""
    return {
        "low": trapezoidal_membership(bp, 0, 0, 90, 110),
        "normal": triangular_membership(bp, 90, 120, 150),
        "high": trapezoidal_membership(bp, 130, 150, 220, 220),
    }


def fuzzify_cholesterol(chol):
    """Find the membership of cholesterol in Normal, Borderline, and High sets."""
    return {
        "normal": trapezoidal_membership(chol, 0, 0, 180, 220),
        "borderline": triangular_membership(chol, 180, 225, 270),
        "high": trapezoidal_membership(chol, 240, 270, 600, 600),
    }


# ---------------------------------------------------------
# 3. OUTPUT MEMBERSHIP FUNCTIONS
# The output represents an illustrative risk score from 0 to 100.
# ---------------------------------------------------------

def risk_membership(score):
    """Calculate membership in Low, Medium, and High output sets."""
    return {
        "low": trapezoidal_membership(score, 0, 0, 20, 45),
        "medium": triangular_membership(score, 25, 50, 75),
        "high": trapezoidal_membership(score, 55, 80, 100, 100),
    }


# ---------------------------------------------------------
# 4. FUZZY RULE BASE
# Apply Mamdani rules using MIN for AND.
# ---------------------------------------------------------

def evaluate_rules(age, bp, chol):
    """
    Evaluate the manually defined fuzzy rules.

    Each rule returns:
        output category and rule activation strength.
    """
    rules = []

    # Rule 1: Older age AND high cholesterol -> High risk
    rules.append((
        "high",
        min(age["older"], chol["high"])
    ))

    # Rule 2: High blood pressure AND high cholesterol -> High risk
    rules.append((
        "high",
        min(bp["high"], chol["high"])
    ))

    # Rule 3: Older age AND high blood pressure -> High risk
    rules.append((
        "high",
        min(age["older"], bp["high"])
    ))

    # Rule 4: Middle age AND borderline cholesterol -> Medium risk
    rules.append((
        "medium",
        min(age["middle"], chol["borderline"])
    ))

    # Rule 5: Young age AND normal BP AND normal cholesterol -> Low risk
    rules.append((
        "low",
        min(age["young"], bp["normal"], chol["normal"])
    ))

    # Rule 6: Normal BP AND normal cholesterol -> Low risk
    rules.append((
        "low",
        min(bp["normal"], chol["normal"])
    ))

    # Rule 7: Older age -> Medium risk
    rules.append(("medium", age["older"]))

    # Rule 8: High blood pressure -> Medium risk
    rules.append(("medium", bp["high"]))

    # Rule 9: High cholesterol -> Medium risk
    rules.append(("medium", chol["high"]))

    return rules


# ---------------------------------------------------------
# 5. MAMDANI INFERENCE AND DEFUZZIFICATION
# ---------------------------------------------------------

def calculate_risk(age_value, bp_value, chol_value):
    """
    Calculate the final fuzzy risk score.

    Steps:
        1. Validate inputs.
        2. Fuzzify the inputs.
        3. Evaluate fuzzy rules.
        4. Aggregate rule outputs using MAX.
        5. Apply centroid defuzzification.
        6. Return the score and category.
    """

    # Basic input validation
    if not 0 <= age_value <= 120:
        raise ValueError("Age must be between 0 and 120.")

    if not 0 <= bp_value <= 300:
        raise ValueError("Blood pressure must be between 0 and 300.")

    if not 0 <= chol_value <= 1000:
        raise ValueError("Cholesterol must be between 0 and 1000.")

    # Step 1: Fuzzification
    age_fuzzy = fuzzify_age(age_value)
    bp_fuzzy = fuzzify_blood_pressure(bp_value)
    chol_fuzzy = fuzzify_cholesterol(chol_value)

    # Step 2: Evaluate all rules
    rules = evaluate_rules(age_fuzzy, bp_fuzzy, chol_fuzzy)

    # Step 3: Create the output universe (0 to 100)
    output_values = np.arange(0, 101, 1)

    # Initially, no output value has any membership.
    aggregated_output = np.zeros(len(output_values))

    # Step 4: Apply each rule's activation strength.
    # Mamdani implication clips the output membership function
    # at the strength of the corresponding rule.
    for category, strength in rules:

        if strength <= 0:
            continue

        for index, score in enumerate(output_values):
            membership = risk_membership(score)[category]

            clipped_membership = min(membership, strength)

            # MAX aggregation combines contributions from rules.
            aggregated_output[index] = max(
                aggregated_output[index],
                clipped_membership
            )

    # Step 5: Centroid defuzzification.
    # It calculates the centre of area of the aggregated fuzzy output.
    total_membership = np.sum(aggregated_output)

    if total_membership == 0:
        # No rule activated, so the system cannot infer a score.
        return {
            "risk_score": None,
            "category": "Undetermined",
            "rules_evaluated": len(rules),
        }

    risk_score = float(
        np.sum(output_values * aggregated_output) / total_membership
    )

    # Step 6: Convert the numeric score into a label.
    if risk_score < 40:
        category = "Low"
    elif risk_score < 65:
        category = "Medium"
    else:
        category = "High"

    return {
        "risk_score": round(risk_score, 2),
        "category": category,
        "rules_evaluated": len(rules),
    }


# ---------------------------------------------------------
# 6. RUN A SAMPLE
# ---------------------------------------------------------

if __name__ == "__main__":

    # Example patient values for demonstration only.
    patient_age = 62
    patient_bp = 150
    patient_cholesterol = 270

    result = calculate_risk(
        patient_age,
        patient_bp,
        patient_cholesterol
    )

    print("\n===== Mamdani Fuzzy Logic Demonstration =====")
    print(f"Age: {patient_age}")
    print(f"Blood pressure: {patient_bp}")
    print(f"Cholesterol: {patient_cholesterol}")

    print(f"\nEstimated score: {result['risk_score']}")
    print(f"Illustrative category: {result['category']}")
    print(f"Rules evaluated: {result['rules_evaluated']}")

    print("\nNote: This score is for an academic demonstration,")
    print("not for clinical assessment or medical decisions.")
