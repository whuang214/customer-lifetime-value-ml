import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, Ridge, LassoCV, RidgeCV, BayesianRidge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# -------------- Helper Functions -------------- #


def load_dataset(data_path):
    """Load the dataset from CSV into a DataFrame."""
    data = pd.read_csv(data_path)
    return data


def prepare_data(data):
    """
    Define the target and predictors, split out X (features) and y (target).
    Returns X, y, and predictor names.
    """
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
    X = data[predictors]
    y = data[target]
    return X, y, predictors


def preprocess_data(X_train, X_val):
    """
    Use OneHotEncoder to transform categorical features, leaving numeric columns as is.
    Returns the transformed training and validation sets plus feature names after encoding.
    """
    # Identify categorical features
    categorical_features = X_train.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(drop="first"), categorical_features)],
        remainder="passthrough",
    )

    X_train_enc = preprocessor.fit_transform(X_train)
    X_val_enc = preprocessor.transform(X_val)

    # Extract the expanded feature names from the encoder
    feature_names = preprocessor.get_feature_names_out()
    return X_train_enc, X_val_enc, feature_names


def evaluate_model(model, X_train, y_train, X_val, y_val):
    """
    Fit the model on the training data, evaluate on both train and validation sets,
    and return the fitted model and coefficient information.
    """
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Compute metrics
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)

    val_r2 = r2_score(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)

    print(f"  Train R2:  {train_r2:.3f}, RMSE: {train_rmse:.3f}, MAE: {train_mae:.3f}")
    print(f"  Valid R2:  {val_r2:.3f}, RMSE: {val_rmse:.3f}, MAE: {val_mae:.3f}")

    return model


def get_nonzero_features(model, feature_names, threshold=1e-5):
    """
    For the given model, return a list of (feature_name, coefficient) for all
    coefficients that are not (approximately) zero.

    Note:
      - For Lasso, many coefficients can be exactly zero.
      - For Ridge/BayesianRidge, you may only find "small" non-zero coefficients,
        so we set a threshold to treat small values as effectively zero.
    """
    coefs = model.coef_
    nonzero_indices = np.where(abs(coefs) > threshold)[0]
    return [(feature_names[idx], coefs[idx]) for idx in nonzero_indices]


def print_top_features(feature_list):
    """
    Given a list of (feature_name, coefficient) pairs, print them sorted
    by absolute coefficient size descending.
    """
    # Sort by absolute value of coefficient
    feature_list_sorted = sorted(feature_list, key=lambda x: abs(x[1]), reverse=True)
    for feat, coef in feature_list_sorted:
        print(f"    {feat}: {coef:.5f}")


# -------------- Main Runner -------------- #


def run_regularization_methods(data_path):
    """
    1) Load and prepare data
    2) Split into train & validation
    3) Preprocess (one-hot)
    4) Fit Lasso, Ridge, LassoCV, RidgeCV, BayesianRidge
    5) Check how many features remain (non-zero) for each approach
    6) Compare to see if the subset of variables is reduced
    """

    # 1. Load dataset
    data = load_dataset(data_path)

    # 2. Prepare X and y
    X, y, predictors = prepare_data(data)

    # 3. Split train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Preprocess / one-hot encode
    X_train_enc, X_val_enc, feature_names = preprocess_data(X_train, X_val)

    # -------------------- (a) Lasso -------------------- #
    print("\n--- (a) Lasso (alpha=1.0) ---")
    lasso_model = Lasso(alpha=1.0, random_state=42)
    lasso_model = evaluate_model(lasso_model, X_train_enc, y_train, X_val_enc, y_val)
    lasso_features = get_nonzero_features(lasso_model, feature_names, threshold=1e-5)
    print(f"  #Non-Zero Features: {len(lasso_features)} of {len(feature_names)}")
    if len(lasso_features) > 0:
        print("  Non-zero Coefficients (sorted by magnitude):")
        print_top_features(lasso_features)

    # -------------------- (b) Ridge -------------------- #
    print("\n--- (b) Ridge (alpha=1.0) ---")
    ridge_model = Ridge(alpha=1.0, random_state=42)
    ridge_model = evaluate_model(ridge_model, X_train_enc, y_train, X_val_enc, y_val)
    ridge_features = get_nonzero_features(ridge_model, feature_names, threshold=1e-3)
    # Adjust threshold for "near-zero" in Ridge since it doesn't produce exact zeros
    print(f"  #Features above threshold: {len(ridge_features)} of {len(feature_names)}")
    if len(ridge_features) > 0:
        print("  Coefficients (sorted by magnitude):")
        print_top_features(ridge_features)

    # -------------------- (c) LassoCV -------------------- #
    print("\n--- (c) LassoCV ---")
    lasso_cv_model = LassoCV(cv=5, random_state=42)
    lasso_cv_model = evaluate_model(
        lasso_cv_model, X_train_enc, y_train, X_val_enc, y_val
    )
    print(f"  Best alpha found by LassoCV: {lasso_cv_model.alpha_:.4f}")
    lasso_cv_features = get_nonzero_features(
        lasso_cv_model, feature_names, threshold=1e-5
    )
    print(f"  #Non-Zero Features: {len(lasso_cv_features)} of {len(feature_names)}")
    if len(lasso_cv_features) > 0:
        print("  Non-zero Coefficients (sorted by magnitude):")
        print_top_features(lasso_cv_features)

    # -------------------- (d) RidgeCV -------------------- #
    print("\n--- (d) RidgeCV ---")
    # By default, RidgeCV tests alphas in [0.1, 1.0, 10.0, 100.0] if not specified
    ridge_cv_model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
    ridge_cv_model = evaluate_model(
        ridge_cv_model, X_train_enc, y_train, X_val_enc, y_val
    )
    print(f"  Best alpha found by RidgeCV: {ridge_cv_model.alpha_:.4f}")
    ridge_cv_features = get_nonzero_features(
        ridge_cv_model, feature_names, threshold=1e-3
    )
    print(
        f"  #Features above threshold: {len(ridge_cv_features)} of {len(feature_names)}"
    )
    if len(ridge_cv_features) > 0:
        print("  Coefficients (sorted by magnitude):")
        print_top_features(ridge_cv_features)

    # -------------------- (e) BayesianRidge -------------------- #
    print("\n--- (e) BayesianRidge ---")
    bayes_ridge_model = BayesianRidge()
    bayes_ridge_model = evaluate_model(
        bayes_ridge_model, X_train_enc, y_train, X_val_enc, y_val
    )
    bayes_ridge_features = get_nonzero_features(
        bayes_ridge_model, feature_names, threshold=1e-3
    )
    print(
        f"  #Features above threshold: {len(bayes_ridge_features)} of {len(feature_names)}"
    )
    if len(bayes_ridge_features) > 0:
        print("  Coefficients (sorted by magnitude):")
        print_top_features(bayes_ridge_features)

    # -------------------------------------------------------- #
    #            Compare results across all methods
    # -------------------------------------------------------- #
    print("\n====================== Conclusion ======================")
    print("Features remaining (non-zero or above threshold) for each:")
    print(f"  Lasso (alpha=1.0):        {len(lasso_features)} / {len(feature_names)}")
    print(f"  Ridge (alpha=1.0):        {len(ridge_features)} / {len(feature_names)}")
    print(
        f"  LassoCV:                  {len(lasso_cv_features)} / {len(feature_names)}"
    )
    print(
        f"  RidgeCV:                  {len(ridge_cv_features)} / {len(feature_names)}"
    )
    print(
        f"  BayesianRidge:            {len(bayes_ridge_features)} / {len(feature_names)}"
    )


if __name__ == "__main__":
    run_regularization_methods("data/CustomerData_Composite-1.csv")
