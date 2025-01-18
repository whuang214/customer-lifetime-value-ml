import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def print_metrics(
    train_r2, train_mae, train_rmse, val_r2, val_mae, val_rmse, coefficients
):
    """
    Print model performance metrics (R2, MAE, RMSE) for both
    training and validation sets, plus the sorted feature coefficients.
    """
    performance_metrics = {
        "Metric": ["R2", "MAE", "RMSE"],
        "Training": [train_r2, train_mae, train_rmse],
        "Validation": [val_r2, val_mae, val_rmse],
    }

    coefficients_df = pd.DataFrame(
        {"Predictor": coefficients.keys(), "Coefficient": coefficients.values()}
    ).sort_values(by="Coefficient", ascending=False)

    print("Model Coefficients:")
    print(coefficients_df)

    print("\nModel Performance Metrics:")
    print(pd.DataFrame(performance_metrics))


def plot_lift_charts(validation_results):
    """
    Plot both the standard Lift Chart and a Decile Lift Chart to visualize
    how well the predicted values capture actual CLTV compared to a random
    or baseline approach.
    """
    # Sort by predicted CLTV in descending order
    validation_results = validation_results.sort_values(by="Predicted", ascending=False)
    validation_results["Cumulative Actual"] = validation_results["Actual"].cumsum()
    validation_results["Cumulative Random"] = np.linspace(
        0, validation_results["Actual"].sum(), len(validation_results)
    )

    # -------------------- Lift Chart --------------------
    plt.figure(figsize=(10, 6))
    plt.plot(
        np.arange(1, len(validation_results) + 1),
        validation_results["Cumulative Actual"],
        label="Model",
    )
    plt.plot(
        np.arange(1, len(validation_results) + 1),
        validation_results["Cumulative Random"],
        label="Random",
    )
    plt.xlabel("Number of Customers")
    plt.ylabel("Cumulative CLTV")
    plt.title("Lift Chart")
    plt.legend()
    plt.grid()
    plt.show()

    # -------------------- Decile Lift Chart --------------------
    # Create 10 bins (deciles) based on predicted values
    validation_results["Decile"] = pd.qcut(
        validation_results["Predicted"], 10, labels=False, duplicates="drop"
    )
    decile_data = (
        validation_results.groupby("Decile")["Actual"].sum().sort_index(ascending=False)
    )

    # Calculate lift relative to the mean CLTV
    lift = decile_data / validation_results["Actual"].mean()

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(lift) + 1), lift)
    plt.xlabel("Decile")
    plt.ylabel("Lift (Factor)")
    plt.title("Decile Lift Chart")
    plt.grid(axis="y")
    plt.show()


def load_dataset(data_path):
    """Load the dataset from CSV into a DataFrame."""
    data = pd.read_csv(data_path)
    return data


def prepare_data(data):
    """
    Define the target and predictors, split out X (features) and y (target).
    Returns X, y, and a list of predictor names (for reference).
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
    Use OneHotEncoder to transform categorical features, leaving numeric columns as-is.
    Returns the transformed training and validation sets plus feature names after encoding.
    """
    # Identify categorical features (object dtype). Adjust if you have boolean columns, etc.
    categorical_features = X_train.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(drop="first"), categorical_features)],
        remainder="passthrough",
    )

    # Fit on the training set and transform both
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    # Extract feature names for reference
    feature_names = preprocessor.get_feature_names_out()
    return X_train_processed, X_val_processed, feature_names


def train_linear_regression(X_train_processed, y_train):
    """
    Train a linear regression model on the preprocessed training data.
    Returns the fitted model.
    """
    model = LinearRegression()
    model.fit(X_train_processed, y_train)
    return model


def evaluate_model(
    model, X_train_processed, y_train, X_val_processed, y_val, feature_names
):
    """
    Use the trained model to predict on both training and validation sets,
    compute metrics, and return a DataFrame with actual vs. predicted for validation.
    """
    # Predictions for training set
    y_train_pred = model.predict(X_train_processed)
    # Predictions for validation set
    y_val_pred = model.predict(X_val_processed)

    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

    val_r2 = r2_score(y_val, y_val_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

    # Extract coefficients mapped to feature names
    coefficients = dict(zip(feature_names, model.coef_))

    # Print metrics and coefficients
    print_metrics(
        train_r2, train_mae, train_rmse, val_r2, val_mae, val_rmse, coefficients
    )

    # Create a validation DataFrame for residuals and future analysis
    residuals = y_val - y_val_pred
    validation_results = pd.DataFrame(
        {"Actual": y_val, "Predicted": y_val_pred, "Residuals": residuals}
    )

    # Print out the first 5 predictions and residuals for validation
    print("\nValidation Predictions and Residuals (first 5 rows):")
    print(validation_results.head())

    return validation_results


def run_linear_regression_with_categoricals(data_path):
    """
    Orchestrates the entire process:
      1. Load the dataset.
      2. Prepare the data (X, y).
      3. Split into training and validation sets.
      4. Preprocess (one-hot encoding for categoricals).
      5. Train a linear regression model.
      6. Evaluate the model, print metrics, and plot lift charts.
    """
    # 1. Load the dataset
    data = load_dataset(data_path)

    # 2. Prepare X, y
    X, y, _ = prepare_data(data)

    # 3. Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Preprocess / encode categorical variables
    X_train_processed, X_val_processed, feature_names = preprocess_data(X_train, X_val)

    # 5. Train the linear regression model
    model = train_linear_regression(X_train_processed, y_train)

    # 6. Evaluate the model
    validation_results = evaluate_model(
        model, X_train_processed, y_train, X_val_processed, y_val, feature_names
    )

    # 7. Plot lift charts
    plot_lift_charts(validation_results)


if __name__ == "__main__":
    run_linear_regression_with_categoricals("data/CustomerData_Composite-1.csv")
