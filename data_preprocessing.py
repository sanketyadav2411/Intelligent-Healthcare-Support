
# data_preprocessing.py

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ==================================================
# CONFIGURATION
# ==================================================

DATA_PATH = Path("data/heart.csv")
RANDOM_STATE = 42

NUMERIC_COLUMNS = [
    "age",
    "trestbps",
    "chol",
    "thalch",
    "oldpeak"
]

BINARY_COLUMNS = [
    "fbs",
    "exang"
]

CATEGORICAL_COLUMNS = [
    "sex",
    "cp",
    "restecg",
    "slope",
    "thal"
]

FEATURE_COLUMNS = (
    NUMERIC_COLUMNS
    + BINARY_COLUMNS
    + CATEGORICAL_COLUMNS
)


# ==================================================
# 1. LOAD DATA
# ==================================================

def load_data():
    """Load and validate the heart disease dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place heart.csv inside the data folder."
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower()

    required_columns = set(FEATURE_COLUMNS + ["num"])
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    print("Original dataset shape:", df.shape)

    return df


# ==================================================
# 2. PREPARE FEATURES AND TARGET
# ==================================================

def prepare_features(df):
    """Create the binary target and clean feature columns."""

    df = df.copy()

    # Convert target to numeric and remove missing targets.
    df["num"] = pd.to_numeric(df["num"], errors="coerce")
    df = df.dropna(subset=["num"])

    # 0 = no disease; values greater than 0 = disease.
    y = (df["num"] > 0).astype(int)

    X = df[FEATURE_COLUMNS].copy()

    # Convert numeric columns safely.
    for col in NUMERIC_COLUMNS + BINARY_COLUMNS:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # In this dataset, zero is treated as missing for
    # cholesterol and resting blood pressure.
    X["chol"] = X["chol"].replace(0, float("nan"))
    X["trestbps"] = X["trestbps"].replace(0, float("nan"))

    # Preserve missing categorical values as a category.
    for col in CATEGORICAL_COLUMNS:
        X[col] = X[col].fillna("Missing").astype(str)

    print("Feature shape before encoding:", X.shape)
    print("\nTarget distribution:")
    print(
        y.value_counts()
        .rename(index={0: "No disease", 1: "Disease present"})
    )

    return X, y


# ==================================================
# 3. SPLIT DATA
# ==================================================

def split_data(X, y):
    """Split data into 70% train, 15% validation, 15% test."""

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test
    )


# ==================================================
# 4. IMPUTE, ENCODE AND SCALE
# ==================================================

def fit_preprocessing(X_train, X_val, X_test):
    """
    Fit preprocessing on training data only.

    Neural-network data is standardized.
    Fuzzy-model data remains in original numeric units.
    """

    # Separate copies for the fuzzy model.
    raw_train = X_train.copy()
    raw_val = X_val.copy()
    raw_test = X_test.copy()

    # ----------------------------------------------
    # A. Impute numerical columns using medians
    # ----------------------------------------------

    numeric_imputer = SimpleImputer(strategy="median")

    train_num = pd.DataFrame(
        numeric_imputer.fit_transform(
            raw_train[NUMERIC_COLUMNS]
        ),
        columns=NUMERIC_COLUMNS,
        index=raw_train.index
    )

    val_num = pd.DataFrame(
        numeric_imputer.transform(
            raw_val[NUMERIC_COLUMNS]
        ),
        columns=NUMERIC_COLUMNS,
        index=raw_val.index
    )

    test_num = pd.DataFrame(
        numeric_imputer.transform(
            raw_test[NUMERIC_COLUMNS]
        ),
        columns=NUMERIC_COLUMNS,
        index=raw_test.index
    )

    # ----------------------------------------------
    # B. Impute binary columns using most frequent
    # ----------------------------------------------

    binary_imputer = SimpleImputer(
        strategy="most_frequent",
        keep_empty_features=True
    )

    train_bin = pd.DataFrame(
        binary_imputer.fit_transform(
            raw_train[BINARY_COLUMNS]
        ),
        columns=BINARY_COLUMNS,
        index=raw_train.index
    )

    val_bin = pd.DataFrame(
        binary_imputer.transform(
            raw_val[BINARY_COLUMNS]
        ),
        columns=BINARY_COLUMNS,
        index=raw_val.index
    )

    test_bin = pd.DataFrame(
        binary_imputer.transform(
            raw_test[BINARY_COLUMNS]
        ),
        columns=BINARY_COLUMNS,
        index=raw_test.index
    )

    # ----------------------------------------------
    # C. One-hot encode categorical columns
    # ----------------------------------------------

    train_cat = pd.get_dummies(
        raw_train[CATEGORICAL_COLUMNS],
        prefix=CATEGORICAL_COLUMNS,
        dtype=int
    )

    val_cat = pd.get_dummies(
        raw_val[CATEGORICAL_COLUMNS],
        prefix=CATEGORICAL_COLUMNS,
        dtype=int
    )

    test_cat = pd.get_dummies(
        raw_test[CATEGORICAL_COLUMNS],
        prefix=CATEGORICAL_COLUMNS,
        dtype=int
    )

    # Use the training columns for every dataset.
    val_cat = val_cat.reindex(
        columns=train_cat.columns,
        fill_value=0
    )

    test_cat = test_cat.reindex(
        columns=train_cat.columns,
        fill_value=0
    )

    # ----------------------------------------------
    # D. Combine features
    # ----------------------------------------------

    train_encoded = pd.concat(
        [train_num, train_bin, train_cat], axis=1
    )

    val_encoded = pd.concat(
        [val_num, val_bin, val_cat], axis=1
    )

    test_encoded = pd.concat(
        [test_num, test_bin, test_cat], axis=1
    )

    # Guarantee identical columns and ordering.
    val_encoded = val_encoded.reindex(
        columns=train_encoded.columns
    )

    test_encoded = test_encoded.reindex(
        columns=train_encoded.columns
    )

    # ----------------------------------------------
    # E. Standardize for the neural network
    # ----------------------------------------------

    scaler = StandardScaler()

    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_encoded),
        columns=train_encoded.columns,
        index=train_encoded.index
    )

    val_scaled = pd.DataFrame(
        scaler.transform(val_encoded),
        columns=train_encoded.columns,
        index=val_encoded.index
    )

    test_scaled = pd.DataFrame(
        scaler.transform(test_encoded),
        columns=train_encoded.columns,
        index=test_encoded.index
    )

    print("\nPreprocessing complete.")
    print("Training rows:", len(train_scaled))
    print("Validation rows:", len(val_scaled))
    print("Test rows:", len(test_scaled))
    print("Encoded feature count:", train_encoded.shape[1])

    # IMPORTANT:
    # Return only features and preprocessing objects here.
    # Target labels are added in load_and_preprocess().
    return {
        "nn": {
            "X_train": train_scaled,
            "X_val": val_scaled,
            "X_test": test_scaled
        },
        "fuzzy": {
            "X_train": train_encoded,
            "X_val": val_encoded,
            "X_test": test_encoded
        },
        "scaler": scaler,
        "numeric_imputer": numeric_imputer,
        "binary_imputer": binary_imputer,
        "feature_columns": list(train_encoded.columns)
    }


# ==================================================
# 5. MAIN PREPROCESSING FUNCTION
# ==================================================

def load_and_preprocess():
    """Prepare data and return features plus target labels."""

    df = load_data()
    X, y = prepare_features(df)

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test
    ) = split_data(X, y)

    # Preprocess features.
    data = fit_preprocessing(
        X_train, X_val, X_test
    )

    # Add target labels here, where they are defined.
    data["y_train"] = y_train
    data["y_val"] = y_val
    data["y_test"] = y_test

    return data


# ==================================================
# 6. TEST THIS FILE
# ==================================================

if __name__ == "__main__":
    data = load_and_preprocess()

    print("\nNeural network sample:")
    print(data["nn"]["X_train"].head())

    print("\nFuzzy model sample:")
    print(data["fuzzy"]["X_train"].head())

    print("\nTarget sample:")
    print(data["y_train"].head())
