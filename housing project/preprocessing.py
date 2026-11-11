"""
preprocessing.py
----------------
Data loading, cleaning, validation, and encoding module for the Housing Price Prediction System.
Ensures identical preprocessing is applied during model training and real-time user predictions.
"""

import os
import pandas as pd
import numpy as np

# Binary columns in the dataset mapped to 1 (yes) and 0 (no)
BINARY_COLUMNS = [
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'prefarea'
]

# Furnishing categories
FURNISHING_CATEGORIES = ['furnished', 'semi-furnished', 'unfurnished']

# Base numerical columns
NUMERICAL_COLUMNS = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']


def load_dataset(filepath="Housing.csv"):
    """
    Load and validate the Housing.csv dataset.
    Returns:
        pd.DataFrame: Cleaned pandas DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    df = pd.read_csv(filepath)

    # Basic validations
    expected_cols = [
        'price', 'area', 'bedrooms', 'bathrooms', 'stories',
        'mainroad', 'guestroom', 'basement', 'hotwaterheating',
        'airconditioning', 'parking', 'prefarea', 'furnishingstatus'
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return df


def preprocess_data(df, drop_target=True):
    """
    Preprocess the raw DataFrame for Machine Learning:
    - Maps 'yes'/'no' binary categorical features to 1/0
    - Encodes 'furnishingstatus' using dummy encoding (drop_first=True, baseline is 'furnished')
      creating: 'furnishingstatus_semi-furnished' and 'furnishingstatus_unfurnished'
    - Prevents multicollinearity (dummy variable trap)

    Args:
        df (pd.DataFrame): Raw housing dataframe
        drop_target (bool): Whether to separate price as target y

    Returns:
        If drop_target is True: (X, y, feature_names)
        If drop_target is False: (X, feature_names)
    """
    df_proc = df.copy()

    # Convert binary categorical variables
    for col in BINARY_COLUMNS:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0}).fillna(0).astype(int)

    # Convert furnishing status into explicit dummy variables to guarantee consistent column names
    # Baseline reference category = 'furnished' (when both dummies are 0)
    if 'furnishingstatus' in df_proc.columns:
        status_clean = df_proc['furnishingstatus'].astype(str).str.strip().str.lower()
        df_proc['furnishingstatus_semi-furnished'] = (status_clean == 'semi-furnished').astype(int)
        df_proc['furnishingstatus_unfurnished'] = (status_clean == 'unfurnished').astype(int)
        df_proc.drop(columns=['furnishingstatus'], inplace=True)

    if drop_target and 'price' in df_proc.columns:
        y = df_proc['price'].astype(float)
        X = df_proc.drop(columns=['price'])
        return X, y, list(X.columns)
    else:
        return df_proc, list(df_proc.columns)


def prepare_user_input(input_dict, feature_columns):
    """
    Converts a dictionary of user input into a single-row DataFrame aligned with training columns.

    Args:
        input_dict (dict): Dictionary with keys:
            area, bedrooms, bathrooms, stories, mainroad ('yes'/'no'),
            guestroom ('yes'/'no'), basement ('yes'/'no'), hotwaterheating ('yes'/'no'),
            airconditioning ('yes'/'no'), parking, prefarea ('yes'/'no'),
            furnishingstatus ('Furnished', 'Semi-furnished', 'Unfurnished')
        feature_columns (list): Exact list of feature column names used by the trained model.

    Returns:
        pd.DataFrame: 1-row DataFrame ready for model prediction.
    """
    raw_df = pd.DataFrame([input_dict])
    proc_df, _ = preprocess_data(raw_df, drop_target=False)

    # Ensure all required features are present (fill missing with 0)
    for col in feature_columns:
        if col not in proc_df.columns:
            proc_df[col] = 0

    # Retain exact column order
    return proc_df[feature_columns].astype(float)


def format_inr(number):
    """
    Format a numeric amount into Indian Currency notation:
    e.g. 7540000 -> ₹ 75,40,000 (₹ 75.40 Lakhs)
    """
    try:
        val = float(number)
    except (ValueError, TypeError):
        return "₹ 0"

    is_negative = val < 0
    val_abs = abs(val)

    # Indian comma formatting
    s = f"{val_abs:.0f}"
    if len(s) > 3:
        last3 = s[-3:]
        remaining = s[:-3]
        groups = []
        while len(remaining) > 2:
            groups.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            groups.insert(0, remaining)
        formatted_num = ",".join(groups) + "," + last3
    else:
        formatted_num = s

    # Also calculate readable denomination (Crores / Lakhs)
    if val_abs >= 10000000:
        cr = val_abs / 10000000.0
        denom = f"{cr:.2f} Cr"
    elif val_abs >= 100000:
        lakh = val_abs / 100000.0
        denom = f"{lakh:.2f} Lakhs"
    elif val_abs >= 1000:
        k = val_abs / 1000.0
        denom = f"{k:.1f} K"
    else:
        denom = f"{val_abs:.0f}"

    prefix = "-₹ " if is_negative else "₹ "
    return f"{prefix}{formatted_num}", denom

