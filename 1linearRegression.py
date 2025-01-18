import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt


def print_metrics(
    train_r2, train_mae, train_rmse, val_r2, val_mae, val_rmse, coefficients
):
    """Print model performance metrics and feature coefficients."""
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
    """Plot lift charts and decile lift charts."""
    validation_results = validation_results.sort_values(by="Predicted", ascending=False)
    validation_results["Cumulative Actual"] = validation_results["Actual"].cumsum()
    validation_results["Cumulative Random"] = np.linspace(
        0, validation_results["Actual"].sum(), len(validation_results)
    )

    # Lift Chart
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

    # Decile Lift Chart
    validation_results["Decile"] = pd.qcut(
        validation_results["Predicted"], 10, labels=False, duplicates="drop"
    )
    decile_data = (
        validation_results.groupby("Decile")["Actual"].sum().sort_index(ascending=False)
    )

    lift = decile_data / validation_results["Actual"].mean()

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(lift) + 1), lift)
    plt.xlabel("Decile")
    plt.ylabel("Lift (Factor)")
    plt.title("Decile Lift Chart")
    plt.grid(axis="y")
    plt.show()


def run_linear_regression_with_categoricals(data_path):
    # Load the dataset
    data = pd.read_csv(data_path)

    # Define the target and predictors
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

    # Prepare the data
    X = data[predictors]
    y = data[target]

    # Handle categorical variables using OneHotEncoder
    categorical_features = X.select_dtypes(include=["object"]).columns
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(drop="first"), categorical_features)],
        remainder="passthrough",
    )

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocess the data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    # Train the model
    model = LinearRegression()
    model.fit(X_train_processed, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train_processed)
    y_val_pred = model.predict(X_val_processed)

    # Calculate residuals for validation set
    residuals = y_val - y_val_pred

    # Combine predictions and residuals into a validation results DataFrame
    validation_results = pd.DataFrame(
        {"Actual": y_val, "Predicted": y_val_pred, "Residuals": residuals}
    )

    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

    val_r2 = r2_score(y_val, y_val_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

    # Extract feature coefficients
    feature_names = preprocessor.get_feature_names_out()
    coefficients = dict(zip(feature_names, model.coef_))

    # Print metrics and coefficients
    print_metrics(
        train_r2, train_mae, train_rmse, val_r2, val_mae, val_rmse, coefficients
    )

    # Display residuals
    print("\nValidation Residuals:")
    print(validation_results.head())

    # Plot lift charts
    plot_lift_charts(validation_results)


# Run the function with your dataset
run_linear_regression_with_categoricals("data/CustomerData_Composite-1.csv")
