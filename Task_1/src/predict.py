"""
Module: predict.py
Author: Devesh Kumawat
Purpose: Load the trained Iris classification model and perform predictions on new inputs.
         Supports both command-line arguments and interactive shell input.
Date: 2026-07-06
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np

# Configure standard streams to use UTF-8 for cross-platform emoji support
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def load_pipeline(model_path: str = "models/best_model.pkl") -> dict:
    """
    Loads the trained model pipeline and metadata.
    
    Args:
        model_path (str): Path to the saved pipeline pickle.
        
    Returns:
        dict: The pipeline dictionary containing model, scaler, label_encoder, etc.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            f"Please run 'python src/train.py' to train the model first."
        )
    return joblib.load(model_path)

def make_prediction(pipeline: dict, features: list) -> tuple:
    """
    Preprocesses the features and makes a prediction using the trained model.
    
    Args:
        pipeline (dict): The loaded pipeline dictionary.
        features (list): A list of 4 float values [SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm].
        
    Returns:
        tuple: (predicted_class_name, confidence, probabilities_dict)
    """
    model = pipeline["model"]
    scaler = pipeline["scaler"]
    label_encoder = pipeline["label_encoder"]
    feature_names = pipeline["feature_names"]
    class_names = pipeline["class_names"]
    
    # Format inputs as DataFrame for the scaler and model to preserve feature name warnings/logic
    features_df = pd.DataFrame([features], columns=feature_names)
    
    # Scale features
    features_scaled = scaler.transform(features_df)
    
    # Predict class
    pred_code = model.predict(features_scaled)[0]
    predicted_class = label_encoder.inverse_transform([pred_code])[0]
    
    # Predict probabilities (if the model supports it)
    probabilities_dict = {}
    confidence = 1.0
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features_scaled)[0]
        confidence = probs[pred_code]
        for name, prob in zip(class_names, probs):
            probabilities_dict[name] = prob
            
    return predicted_class, confidence, probabilities_dict

def run_interactive(pipeline: dict):
    """Runs an interactive command-line session to predict Iris species."""
    print("==================================================")
    print("   🌸 Iris Flower Classifier - Interactive Mode 🌸")
    print("==================================================")
    print(f"Loaded Model: {pipeline['model_name']}")
    print(f"Model Performance - Accuracy: {pipeline['performance']['accuracy']:.2f}% | F1-Score: {pipeline['performance']['f1_score']:.2%}")
    print("--------------------------------------------------")
    print("Enter the flower measurements in centimeters:")
    
    try:
        sepal_length = float(input(" -> Sepal Length (e.g. 5.1): "))
        sepal_width = float(input(" -> Sepal Width (e.g. 3.5): "))
        petal_length = float(input(" -> Petal Length (e.g. 1.4): "))
        petal_width = float(input(" -> Petal Width (e.g. 0.2): "))
        
        inputs = [sepal_length, sepal_width, petal_length, petal_width]
        pred_class, conf, probs = make_prediction(pipeline, inputs)
        
        print("\n---------------- RESULTS ----------------")
        print(f"predicted Species : 🏆 {pred_class} (Confidence: {conf:.2%})")
        if probs:
            print("Class Probabilities:")
            for name, prob in probs.items():
                marker = "⭐ " if name == pred_class else "   "
                print(f"  {marker}{name:16}: {prob:.2%}")
        print("==================================================")
    except ValueError:
        print("\n❌ Error: Please enter valid numeric values for measurements.")
    except Exception as e:
        print(f"\n❌ Error occurred during prediction: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Predict Iris species based on sepal and petal measurements."
    )
    parser.add_argument("--sepal-length", type=float, help="Sepal length in cm")
    parser.add_argument("--sepal-width", type=float, help="Sepal width in cm")
    parser.add_argument("--petal-length", type=float, help="Petal length in cm")
    parser.add_argument("--petal-width", type=float, help="Petal width in cm")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive CLI mode")
    
    args = parser.parse_args()
    
    try:
        pipeline = load_pipeline()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return
        
    # Check if any argument is provided, or if interactive flag is set
    has_args = (
        args.sepal_length is not None or
        args.sepal_width is not None or
        args.petal_length is not None or
        args.petal_width is not None
    )
    
    if args.interactive or not has_args:
        run_interactive(pipeline)
    else:
        # Validate that all 4 measurements are provided
        if None in [args.sepal_length, args.sepal_width, args.petal_length, args.petal_width]:
            print("\n❌ Error: To make a prediction via command-line arguments, you must provide all four measurements:")
            print("  --sepal-length, --sepal-width, --petal-length, and --petal-width")
            return
            
        inputs = [args.sepal_length, args.sepal_width, args.petal_length, args.petal_width]
        pred_class, conf, probs = make_prediction(pipeline, inputs)
        
        print(f"\nPredicted Species: {pred_class} (Confidence: {conf:.2%})")
        if probs:
            print("Probabilities:")
            for name, prob in probs.items():
                print(f"  - {name}: {prob:.2%}")

if __name__ == "__main__":
    main()
