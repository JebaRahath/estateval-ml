"""
model.py
--------
Machine Learning pipeline for the Housing Price Prediction System:
- Train-test splitting with fixed random state (prevents data leakage)
- Multiple Linear Regression (Baseline)
- Recursive Feature Elimination (RFE) via scikit-learn
- Backward Elimination using statsmodels OLS (p-value threshold = 0.05)
- Model evaluation: R², Adjusted R², MAE, MSE, RMSE
- Residual analysis and feature coefficient interpretation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import statsmodels.api as sm


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    Fixed random_state ensures reproducible results across sessions.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def calculate_metrics(y_true, y_pred, n_features):
    """
    Compute regression evaluation metrics:
    - R² Score
    - Adjusted R² Score
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    """
    n = len(y_true)
    p = n_features
    r2 = r2_score(y_true, y_pred)
    
    # Adjusted R-squared
    if n > p + 1:
        r2_adj = 1.0 - ((1.0 - r2) * (n - 1) / (n - p - 1))
    else:
        r2_adj = r2

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return {
        "r2": float(r2),
        "r2_adj": float(r2_adj),
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
    }


def train_baseline_model(X_train, y_train, X_test, y_test):
    """
    Train a baseline Multiple Linear Regression model with ALL features.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics_train = calculate_metrics(y_train, y_train_pred, X_train.shape[1])
    metrics_test = calculate_metrics(y_test, y_test_pred, X_train.shape[1])

    return {
        "model": model,
        "features": list(X_train.columns),
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
        "y_test_pred": y_test_pred,
        "residuals": y_test - y_test_pred,
    }


def perform_rfe(X_train, y_train, X_test, y_test, n_features=10):
    """
    Perform Recursive Feature Elimination (RFE) using scikit-learn.
    Features are scaled on training data only to avoid scale distortion
    (e.g., area in thousands vs binary flags), ensuring fair ranking.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    estimator = LinearRegression()
    rfe = RFE(estimator=estimator, n_features_to_select=n_features)
    rfe.fit(X_train_scaled, y_train)

    selected_features = [col for col, sel in zip(X_train.columns, rfe.support_) if sel]
    ranking_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Selected": rfe.support_,
        "Rank": rfe.ranking_
    }).sort_values(by="Rank")

    # Fit unscaled Linear Regression on selected features for interpretability
    final_rfe_model = LinearRegression()
    final_rfe_model.fit(X_train[selected_features], y_train)

    y_test_pred = final_rfe_model.predict(X_test[selected_features])
    metrics_test = calculate_metrics(y_test, y_test_pred, len(selected_features))

    return {
        "model": final_rfe_model,
        "selected_features": selected_features,
        "ranking_df": ranking_df,
        "metrics_test": metrics_test,
        "y_test_pred": y_test_pred,
        "residuals": y_test - y_test_pred,
    }


def perform_backward_elimination(X_train, y_train, X_test, y_test, p_threshold=0.05):
    """
    Perform Backward Elimination using statsmodels OLS regression.
    Iteratively removes the feature with the highest p-value until all remaining
    predictors have p-value <= p_threshold.
    
    Includes the constant intercept term and logs every elimination step.
    """
    features = list(X_train.columns)
    step = 0
    step_log = []

    while len(features) > 0:
        # Add intercept constant
        X_with_const = sm.add_constant(X_train[features].astype(float))
        ols_model = sm.OLS(y_train.astype(float), X_with_const).fit()

        # Extract p-values (exclude constant)
        p_values = ols_model.pvalues.drop("const", errors="ignore")
        if len(p_values) == 0:
            break

        max_p_value = p_values.max()
        max_p_feature = p_values.idxmax()

        step_info = {
            "Step": step + 1,
            "Total_Features": len(features),
            "Highest_P_Feature": max_p_feature,
            "Highest_P_Value": float(max_p_value),
            "AIC": float(ols_model.aic),
            "Adj_R2": float(ols_model.rsquared_adj),
            "Action": "Removed" if max_p_value > p_threshold else "Kept (Termination Threshold Reached)"
        }
        step_log.append(step_info)

        if max_p_value > p_threshold:
            features.remove(max_p_feature)
            step += 1
        else:
            break

    # Fit final OLS model with constant
    X_train_final = sm.add_constant(X_train[features].astype(float))
    final_ols = sm.OLS(y_train.astype(float), X_train_final).fit()

    # Train equivalent scikit-learn model for straightforward user input prediction
    final_sklearn_model = LinearRegression()
    final_sklearn_model.fit(X_train[features], y_train)

    y_test_pred = final_sklearn_model.predict(X_test[features])
    metrics_test = calculate_metrics(y_test, y_test_pred, len(features))

    return {
        "selected_features": features,
        "ols_model": final_ols,
        "sklearn_model": final_sklearn_model,
        "step_log": pd.DataFrame(step_log),
        "metrics_test": metrics_test,
        "y_test_pred": y_test_pred,
        "residuals": y_test - y_test_pred,
    }


def get_feature_coefficients(model, feature_names):
    """
    Extract readable regression coefficients and format their real-world impact.
    """
    coefs = model.coef_
    intercept = model.intercept_

    df_coef = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefs,
        "Abs_Impact": np.abs(coefs)
    }).sort_values(by="Abs_Impact", ascending=False)

    return df_coef, intercept


def train_complete_pipeline(df):
    """
    High-level orchestrator that runs the entire machine learning pipeline:
    1. Preprocesses dataset
    2. Splits 80/20 train/test
    3. Fits Baseline Multiple Linear Regression
    4. Runs Recursive Feature Elimination (RFE)
    5. Runs Backward Elimination (OLS p < 0.05)
    6. Returns all models, metrics, and dataset splits in a single structured dictionary.
    """
    from preprocessing import preprocess_data

    X, y, all_feature_names = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    # 1. Baseline Model
    baseline = train_baseline_model(X_train, y_train, X_test, y_test)

    # 2. RFE Model
    rfe_res = perform_rfe(X_train, y_train, X_test, y_test, n_features=10)

    # 3. Backward Elimination Model (Primary Academic Model)
    be_res = perform_backward_elimination(X_train, y_train, X_test, y_test, p_threshold=0.05)

    # Final production model used for prediction is the Backward Elimination model
    production_model = be_res["sklearn_model"]
    production_features = be_res["selected_features"]

    return {
        "raw_df": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "all_feature_names": all_feature_names,
        "baseline": baseline,
        "rfe": rfe_res,
        "backward_elimination": be_res,
        "production_model": production_model,
        "production_features": production_features,
    }

