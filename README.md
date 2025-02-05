# Customer Lifetime Value (CLTV) Prediction

## Overview

This repository contains a data science project focused on predicting **Customer Lifetime Value (CLTV)** using various **regression models** and **feature selection techniques**. The project includes **data preprocessing, model training, and evaluation** to determine key drivers of CLTV.

## Project Structure

- **`1linearRegression.py`** – Implements a **Linear Regression Model** for baseline CLTV prediction, with feature importance extraction and visualization.
- **`2predictorEliminator.py`** – Applies **feature selection techniques** (Forward Selection, Backward Elimination, Exhaustive Search) to optimize model performance.
- **`3regulationModel.py`** – Implements **Lasso, Ridge, and Bayesian Ridge Regression** to improve model generalization and prevent overfitting.
- **`CustomerData_Composite.csv`** – Sample dataset containing customer demographic, service, and billing information.

## Features & Methodologies

### 1. Linear Regression Model

- Trains a **Linear Regression Model** to predict **CLTV** based on customer attributes.
- Extracts **feature importance** to determine the most influential predictors.
- Evaluates performance using **R², MAE, and RMSE**.
- Implements **Lift Charts & Decile Analysis** to assess model effectiveness.

### 2. Feature Selection

- Uses multiple selection techniques to optimize feature subsets:
  - **Exhaustive Feature Search** (tests all feature combinations)
  - **Forward Selection** (adds best features stepwise)
  - **Backward Elimination** (removes least useful features)
  - **Stepwise Regression** (combines forward/backward methods)
- Validates performance using **cross-validation & Mean Squared Error (MSE)**.

### 3. Regularized Regression (Lasso, Ridge, Bayesian Ridge)

- Trains **Lasso Regression** to shrink less relevant coefficients to zero.
- Implements **Ridge Regression** to retain all features while reducing coefficient magnitude.
- Uses **LassoCV & RidgeCV** for automatic hyperparameter tuning.
- Analyzes **non-zero coefficients** to extract key business insights.


## Installation & Usage
### Prerequisites
- Python 3.x
- Required Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `mlxtend`

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. Install dependencies:
    ```sh 
    pip install -r requirements.txt
    ```

3. Run the scripts:
    ```sh
    python 1linearRegression.py
    python 2predictorEliminator.py
    python 3regulationModel.py
    ```

## Results & Insights

### Model Performance Metrics

- **Linear Regression**:
  - Training R²: 0.161
  - Validation R²: 0.152
  - MAE: ~905
  - RMSE: ~1081
- **Feature Importance (Top Predictors for CLTV)**:
  - Online Security & Phone Service positively impact CLTV.
  - Device Protection and One-Year Contracts negatively impact CLTV.

### Lift Chart Analysis

- The model **performs better than random selection** but has room for improvement.
- The **Decile Lift Chart** suggests slight misranking, with Decile 2 outperforming Decile 1.

### Feature Selection Findings

- **Exhaustive Search** identified **contract type, device protection, number of dependents, tenure, and total long-distance charges** as key predictors.
- **Stepwise Regression** refined the model further by iteratively adding/removing features for the best performance.

### Regularization Insights

- **Lasso Regression** reduced the model to 22 key features, removing unnecessary ones.
- **Ridge & Bayesian Ridge** maintained all features but shrunk coefficient magnitudes for better generalization.
- **LassoCV & RidgeCV** found optimal penalty values, with LassoCV reducing the feature set to just 3 predictors.
