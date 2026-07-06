"""
Module: data_processing.py
Author: Antigravity AI
Purpose: Data loading, cleaning, scaling, and preprocessing functions for the Iris dataset.
Date: 2026-07-06
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the Iris dataset from a CSV file.
    
    Args:
        file_path (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")
        
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df: pd.DataFrame):
    """
    Cleans the Iris dataset, encodes target labels, and splits features and target.
    
    Args:
        df (pd.DataFrame): Raw dataframe loaded from CSV.
        
    Returns:
        tuple: (X, y, label_encoder)
            - X: DataFrame of features (SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm)
            - y: Series of encoded target labels
            - label_encoder: Fitted LabelEncoder instance for decoding later
    """
    # Create a copy to prevent modifying the original dataframe
    data = df.copy()
    
    # Drop the Id column if it exists as it is just an index
    if 'Id' in data.columns:
        data = data.drop(columns=['Id'])
        
    # Check for missing values and handle them (though Iris is usually complete)
    missing_count = data.isnull().sum().sum()
    if missing_count > 0:
        # Simple imputation for robustness
        data = data.ffill().bfill()
        
    # Split features and target
    X = data.drop(columns=['Species'])
    y = data['Species']
    
    # Encode target labels (Iris-setosa -> 0, Iris-versicolor -> 1, Iris-virginica -> 2)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    return X, y_encoded, label_encoder

def split_and_scale_data(X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    """
    Splits the data into train and test sets and scales the features.
    
    Args:
        X (pd.DataFrame): Features.
        y (np.ndarray): Encoded labels.
        test_size (float): Proportion of dataset to include in the test split.
        random_state (int): Seed used by the random number generator.
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
            - X_train_scaled: Scaled training features.
            - X_test_scaled: Scaled testing features.
            - y_train: Training labels.
            - y_test: Testing labels.
            - scaler: Fitted StandardScaler instance.
    """
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    # Self-test code
    print("Testing data_processing module...")
    try:
        df = load_data("Iris.csv")
        print(f"Data shape: {df.shape}")
        X, y, le = preprocess_data(df)
        print(f"Classes found: {list(le.classes_)}")
        X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
        print("Data processing pipeline successful!")
        print(f"Train features shape: {X_train.shape}, Test features shape: {X_test.shape}")
    except Exception as e:
        print(f"Self-test failed: {e}")
