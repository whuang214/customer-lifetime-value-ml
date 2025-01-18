import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error

from mlxtend.feature_selection import (
    ExhaustiveFeatureSelector,
    SequentialFeatureSelector,
)


def exhaustive_search_selection(X_train, y_train, feature_names):
    """
    Perform an Exhaustive Search (testing all subsets of features) to find
    the best feature subset using cross-validation.

    Returns:
        best_subset_indices (list): Indices of the best feature subset.
    """
    print("Running Exhaustive Search...")
    lr = LinearRegression()

    exhaustive_selector = ExhaustiveFeatureSelector(
        estimator=lr,
        min_features=1,
        max_features=5,  # ADJUST THIS IF NECESSARY
        scoring="neg_mean_squared_error",
        cv=5,
        print_progress=True,
        n_jobs=-1,
    )
    exhaustive_selector = exhaustive_selector.fit(X_train, y_train)

    best_features_exhaustive = list(exhaustive_selector.best_idx_)
    print("\nBest subset of features (Exhaustive Search):")
    for idx in best_features_exhaustive:
        print("  ", feature_names[idx])

    return best_features_exhaustive


def forward_selection(X_train, y_train, feature_names):
    """
    Perform a Sequential Forward Selection to find the best subset of features.
    Each step adds the feature that best improves performance.

    Returns:
        best_subset_indices (list): Indices of the best feature subset.
    """
    print("\nRunning Forward Selection...")
    lr = LinearRegression()

    sfs_forward = SequentialFeatureSelector(
        estimator=lr,
        k_features="best",
        forward=True,
        floating=False,  # pure forward selection
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=-1,
    )
    sfs_forward = sfs_forward.fit(X_train, y_train)

    best_features_forward = list(sfs_forward.k_feature_idx_)
    print("Best subset of features (Forward Selection):")
    for idx in best_features_forward:
        print("  ", feature_names[idx])

    return best_features_forward


def backward_elimination(X_train, y_train, feature_names):
    """
    Perform a Sequential Backward Elimination to find the best subset of features.
    Each step removes the feature that least hurts performance when taken away.

    Returns:
        best_subset_indices (list): Indices of the best feature subset.
    """
    print("\nRunning Backward Elimination...")
    lr = LinearRegression()

    sfs_backward = SequentialFeatureSelector(
        estimator=lr,
        k_features="best",
        forward=False,
        floating=False,  # pure backward elimination
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=-1,
    )
    sfs_backward = sfs_backward.fit(X_train, y_train)

    best_features_backward = list(sfs_backward.k_feature_idx_)
    print("Best subset of features (Backward Elimination):")
    for idx in best_features_backward:
        print("  ", feature_names[idx])

    return best_features_backward


def stepwise_regression(X_train, y_train, feature_names):
    """
    Perform a 'Stepwise' Regression using 'floating' Forward Selection.
    (i.e., it can add and remove features in a forward direction).

    Returns:
        best_subset_indices (list): Indices of the best feature subset.
    """
    print("\nRunning Stepwise (Floating) Regression...")
    lr = LinearRegression()

    sfs_stepwise = SequentialFeatureSelector(
        estimator=lr,
        k_features="best",
        forward=True,
        floating=True,  # allows for adding/dropping in the forward approach
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=-1,
    )
    sfs_stepwise = sfs_stepwise.fit(X_train, y_train)

    best_features_stepwise = list(sfs_stepwise.k_feature_idx_)
    print("Best subset of features (Stepwise/Floating Selection):")
    for idx in best_features_stepwise:
        print("  ", feature_names[idx])

    return best_features_stepwise


def run_feature_selection_methods(data_path):
    # -------------------------------------------------------------------------
    # 1. Load dataset
    # -------------------------------------------------------------------------
    data = pd.read_csv(data_path)

    # Define target and predictors (from your previous script)
    target = "cltv"
    predictors = [
        "number_of_dependents",
        "married",
        "phone_service",
        "internet_service",
        "online_security",
        "online_backup",
        "device_protection",
        "premium_tech_support",
        "streaming_tv",
        "streaming_movies",
        "streaming_music",
        "contract",
        "paperless_billing",
        "payment_method",
        "monthly_ charges",
        "total_long_distance_charges",
        "tenure",
        "avg_monthly_gb_download",
        "unlimited_data",
        "satisfaction_score",
        "number_of_referrals",
    ]

    # -------------------------------------------------------------------------
    # 2. Prepare X and y
    # -------------------------------------------------------------------------
    X = data[predictors]
    y = data[target]

    # -------------------------------------------------------------------------
    # 3. Preprocess categorical variables using ColumnTransformer
    # -------------------------------------------------------------------------
    categorical_features = X.select_dtypes(include=["object", "bool"]).columns
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(drop="first"), categorical_features)],
        remainder="passthrough",
    )

    X_transformed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()

    # -------------------------------------------------------------------------
    # 4. Train/Validation Split
    # -------------------------------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X_transformed, y, test_size=0.2, random_state=42
    )

    # -------------------------------------------------------------------------
    # 5. Call each feature selection method
    # -------------------------------------------------------------------------
    best_exhaustive = exhaustive_search_selection(X_train, y_train, feature_names)
    best_forward = forward_selection(X_train, y_train, feature_names)
    best_backward = backward_elimination(X_train, y_train, feature_names)
    best_stepwise = stepwise_regression(X_train, y_train, feature_names)

    # -------------------------------------------------------------------------
    # 6. Compare Results on Validation Set
    # -------------------------------------------------------------------------
    print("\n----- Validation Performance Comparison -----")

    def fit_and_evaluate(feature_indices, label):
        lr_model = LinearRegression()
        lr_model.fit(X_train[:, feature_indices], y_train)
        y_pred_val = lr_model.predict(X_val[:, feature_indices])
        mse_val = mean_squared_error(y_val, y_pred_val)
        print(
            f"{label} - #Features: {len(feature_indices)} - Validation MSE: {mse_val:.3f}"
        )

    fit_and_evaluate(best_exhaustive, "Exhaustive Search Subset")
    fit_and_evaluate(best_forward, "Forward Selection Subset")
    fit_and_evaluate(best_backward, "Backward Elimination Subset")
    fit_and_evaluate(best_stepwise, "Stepwise/Floating Subset")


if __name__ == "__main__":
    run_feature_selection_methods("data/CustomerData_Composite-1.csv")
