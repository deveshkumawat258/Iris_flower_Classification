"""
Module: train.py
Author: Antigravity AI
Purpose: Train and evaluate multiple Machine Learning classifiers on the Iris dataset.
         Performs hyperparameter tuning, saves performance figures, and serializes the best model.
Date: 2026-07-06
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure standard streams to use UTF-8 for cross-platform emoji support
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Scikit-learn imports
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Custom data processing functions
from data_processing import load_data, preprocess_data, split_and_scale_data

def setup_directories():
    """Creates the directories for saving models and reports if they don't exist."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    print("Setup directories: 'models/' and 'reports/figures/' are ready.")

def run_eda(df: pd.DataFrame):
    """
    Generates and saves exploratory data analysis plots.
    
    Args:
        df (pd.DataFrame): Raw Iris dataset.
    """
    print("\n--- Running Exploratory Data Analysis (EDA) ---")
    
    # 1. Pairplot
    plot_df = df.copy()
    if 'Id' in plot_df.columns:
        plot_df = plot_df.drop(columns=['Id'])
        
    plt.figure(figsize=(10, 8))
    sns.pairplot(plot_df, hue='Species', palette='Set2', diag_kind='kde')
    plt.gcf().suptitle("Iris Dataset Pairplot", y=1.02, fontsize=16)
    pairplot_path = "reports/figures/pairplot.png"
    plt.savefig(pairplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved pairplot to: {pairplot_path}")
    
    # 2. Correlation Heatmap
    numeric_df = plot_df.drop(columns=['Species'], errors='ignore')
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of Iris Features", fontsize=14, pad=15)
    heatmap_path = "reports/figures/correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation heatmap to: {heatmap_path}")

def train_and_evaluate():
    """
    Main training and evaluation routine.
    Loads data, trains multiple models with grid search, compares performance,
    and serializes the best model.
    """
    setup_directories()
    
    # Load dataset
    dataset_path = "Iris.csv"
    print(f"Loading dataset from {dataset_path}...")
    df = load_data(dataset_path)
    
    # Run EDA
    run_eda(df)
    
    # Preprocess
    X, y, label_encoder = preprocess_data(df)
    feature_names = list(X.columns)
    class_names = list(label_encoder.classes_)
    
    # Split and Scale
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
    
    # Define models and their hyperparameter grids
    models_config = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "grid": {
                "C": [0.1, 1.0, 10.0, 100.0],
                "solver": ["lbfgs", "liblinear"]
            }
        },
        "K-Nearest Neighbors": {
            "model": KNeighborsClassifier(),
            "grid": {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"]
            }
        },
        "Support Vector Machine": {
            "model": SVC(probability=True, random_state=42),
            "grid": {
                "C": [0.1, 1.0, 10.0, 100.0],
                "kernel": ["linear", "rbf", "poly"],
                "gamma": ["scale", "auto"]
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 3, 5, 8],
                "min_samples_split": [2, 5, 10]
            }
        }
    }
    
    best_models = {}
    performance_summary = []
    
    print("\n--- Training Models with Hyperparameter Tuning (5-Fold CV) ---")
    
    for name, config in models_config.items():
        print(f"Tuning {name}...")
        grid_search = GridSearchCV(
            estimator=config["model"],
            param_grid=config["grid"],
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        best_clf = grid_search.best_estimator_
        best_models[name] = best_clf
        
        # Predict and evaluate on test set
        y_pred = best_clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        
        performance_summary.append({
            "Model": name,
            "Best Params": str(grid_search.best_params_),
            "Test Accuracy": acc,
            "Precision (Weighted)": precision,
            "Recall (Weighted)": recall,
            "F1-Score (Weighted)": f1
        })
        
        print(f"  -> Best Params: {grid_search.best_params_}")
        print(f"  -> Test Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")
        
    # Convert summary to DataFrame for display
    summary_df = pd.DataFrame(performance_summary)
    print("\n--- Model Performance Comparison ---")
    print(summary_df.to_string(index=False))
    
    # Select best model based on F1-Score
    best_row = summary_df.loc[summary_df["F1-Score (Weighted)"].idxmax()]
    best_model_name = best_row["Model"]
    best_model_instance = best_models[best_model_name]
    
    print(f"\n🏆 Best Model Selected: {best_model_name} (F1-Score: {best_row['F1-Score (Weighted)']:.4f})")
    
    # --- PCA Dimensionality Reduction ---
    print("\n--- Running PCA Dimensionality Reduction ---")
    
    # Combine train + test scaled data for full-dataset PCA visualization
    X_all_scaled = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_all_scaled)
    
    explained = pca.explained_variance_ratio_ * 100
    print(f"PCA Variance Explained: PC1={explained[0]:.1f}%, PC2={explained[1]:.1f}%")
    
    # Build per-class coordinate lists for the web app
    pca_clusters = {}
    for code, name in enumerate(class_names):
        mask = (y_all == code)
        pca_clusters[name] = {
            "x": X_pca[mask, 0].tolist(),
            "y": X_pca[mask, 1].tolist()
        }
    
    # Generate static PCA scatter plot
    colors = ["#00f5d4", "#f72585", "#f8961e"]
    plt.figure(figsize=(10, 7), facecolor="#0d0d1a")
    ax = plt.gca()
    ax.set_facecolor("#0d0d1a")
    for code, name in enumerate(class_names):
        mask = (y_all == code)
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            label=name, color=colors[code],
            alpha=0.85, edgecolors="white", linewidths=0.4, s=80
        )
    ax.set_title(
        f"PCA Cluster Visualization (PC1={explained[0]:.1f}%, PC2={explained[1]:.1f}%)",
        color="white", fontsize=14, pad=15
    )
    ax.set_xlabel("Principal Component 1", color="#aaaacc")
    ax.set_ylabel("Principal Component 2", color="#aaaacc")
    ax.tick_params(colors="#aaaacc")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    legend = ax.legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#333355")
    plt.tight_layout()
    pca_path = "reports/figures/pca_clusters.png"
    plt.savefig(pca_path, dpi=300, bbox_inches='tight', facecolor="#0d0d1a")
    plt.close()
    print(f"Saved PCA cluster plot to: {pca_path}")
    
    # Save the complete pipeline/metadata as a single joblib file
    model_save_path = "models/best_model.pkl"
    pipeline_data = {
        "model_name": best_model_name,
        "model": best_model_instance,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_names": feature_names,
        "class_names": class_names,
        "pca": pca,
        "pca_clusters": pca_clusters,
        "pca_explained": explained.tolist(),
        "performance": {
            "accuracy": best_row["Test Accuracy"],
            "f1_score": best_row["F1-Score (Weighted)"]
        }
    }
    joblib.dump(pipeline_data, model_save_path)
    print(f"Successfully saved best model pipeline to: {model_save_path}")
    
    # Evaluate best model in detail
    y_pred_best = best_model_instance.predict(X_test)
    print("\n--- Detailed Classification Report (Best Model) ---")
    print(classification_report(y_test, y_pred_best, target_names=class_names))
    
    # Generate and save Confusion Matrix Plot for Best Model
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - Best Model ({best_model_name})", fontsize=14, pad=15)
    cm_path = "reports/figures/confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")
    
    # Generate Feature Importance / Coefficient plot if applicable
    plt.figure(figsize=(10, 6))
    has_importance = False
    
    if hasattr(best_model_instance, "feature_importances_"):
        importances = best_model_instance.feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_features = [feature_names[i] for i in indices]
        sns.barplot(x=importances[indices], y=sorted_features, palette="viridis")
        plt.title(f"Feature Importance - {best_model_name}", fontsize=14)
        plt.xlabel("Importance Score")
        has_importance = True
    elif hasattr(best_model_instance, "coef_"):
        coefs = best_model_instance.coef_
        # If multiclass logistic regression, coef_ is shape (n_classes, n_features)
        if coefs.ndim > 1:
            # Take mean absolute coefficients across classes
            mean_coefs = np.mean(np.abs(coefs), axis=0)
            indices = np.argsort(mean_coefs)[::-1]
            sorted_features = [feature_names[i] for i in indices]
            sns.barplot(x=mean_coefs[indices], y=sorted_features, palette="viridis")
            plt.title(f"Mean Absolute Coefficients (Feature Importance) - {best_model_name}", fontsize=14)
            plt.xlabel("Mean |Coefficient|")
            has_importance = True
            
    if has_importance:
        importance_path = "reports/figures/feature_importance.png"
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved feature importance plot to: {importance_path}")
    else:
        print("Feature importance plotting not applicable for the selected model.")

if __name__ == "__main__":
    train_and_evaluate()
