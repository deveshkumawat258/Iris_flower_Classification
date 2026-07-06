# 🌸 Iris Flower Classification Pipeline

A modular, professional machine learning pipeline to classify Iris flowers into their correct species (`Iris-setosa`, `Iris-versicolor`, `Iris-virginica`) based on four sepal and petal measurements.

This project implements hyperparameter tuning, model comparison across multiple algorithms, automated evaluation, visual report generation, and an interactive prediction CLI.

---

## 📁 Project Structure

```text
CodeAlpha/
└── Task_1/
    ├── Iris.csv                  # The Iris flower dataset
    ├── requirements.txt          # Python package requirements
    ├── README.md                 # Project documentation (this file)
    ├── src/
    │   ├── data_processing.py    # Data loading, cleaning, scaling & splitting
    │   ├── train.py              # Model tuning, training, evaluation & plot saving
    │   └── predict.py            # CLI/Interactive prediction tool for inference
    ├── models/
    │   └── best_model.pkl        # Serialized best model pipeline (fitted scaler + model + metadata)
    └── reports/
        └── figures/              # Saved visualizations
            ├── pairplot.png
            ├── correlation_heatmap.png
            ├── confusion_matrix.png
            └── feature_importance.png
```

---

## 🚀 Setup & Installation

Ensure you have **Python 3.8+** installed. Then, follow these steps:

1. Clone or navigate to the project directory:
   ```bash
   cd CodeAlpha/Task_1
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏋️ Training and Evaluation

To preprocess the data, train multiple candidate models with hyperparameter tuning, generate analysis plots, and serialize the best-performing model, run:

```bash
python src/train.py
```

### What happens during training:
1. **Exploratory Data Analysis**: Generates features pairplots and correlation heatmaps under `reports/figures/`.
2. **Hyperparameter Tuning**: Performs a 5-fold cross-validated grid search for:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)
   - Random Forest
3. **Model Selection**: Automatically selects the best model based on F1-Score.
4. **Performance Evaluation**: Logs validation metrics (Accuracy, Precision, Recall, F1) and saves the confusion matrix and feature importance plots.
5. **Model Serialization**: Saves the entire inference pipeline (model, scaler, encoder, metadata) to `models/best_model.pkl`.

---

## 🔮 Predicting New Flowers

You can run predictions on new flower measurements in two ways:

### 1. Interactive CLI Mode (Default)
Run the script without arguments to open an interactive prompt:
```bash
python src/predict.py
```
**Example prompt:**
```text
==================================================
   🌸 Iris Flower Classifier - Interactive Mode 🌸
==================================================
Loaded Model: Support Vector Machine
Model Performance - Accuracy: 100.00% | F1-Score: 100.00%
--------------------------------------------------
Enter the flower measurements in centimeters:
 -> Sepal Length (e.g. 5.1): 5.1
 -> Sepal Width (e.g. 3.5): 3.5
 -> Petal Length (e.g. 1.4): 1.4
 -> Petal Width (e.g. 0.2): 0.2

---------------- RESULTS ----------------
predicted Species : 🏆 Iris-setosa (Confidence: 98.71%)
Class Probabilities:
  ⭐ Iris-setosa     : 98.71%
     Iris-versicolor : 0.95%
     Iris-virginica  : 0.34%
==================================================
```

### 2. Command-Line Arguments Mode
Provide specific measurements directly via flags:
```bash
python src/predict.py --sepal-length 6.2 --sepal-width 3.4 --petal-length 5.4 --petal-width 2.3
```

---

## 📊 Model Performance Results

After running hyperparameter tuning and cross-validation across multiple algorithms, here is the model performance comparison on the test set:

| Model | Best Parameters | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **🏆 Logistic Regression** | `{'C': 10.0, 'solver': 'lbfgs'}` | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| **Random Forest** | `{'max_depth': 3, 'min_samples_split': 2, 'n_estimators': 50}` | 96.67% | 96.97% | 96.67% | 96.66% |
| **Support Vector Machine (SVM)** | `{'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'}` | 93.33% | 93.33% | 93.33% | 93.33% |
| **K-Nearest Neighbors (KNN)** | `{'metric': 'euclidean', 'n_neighbors': 5, 'weights': 'uniform'}` | 93.33% | 94.44% | 93.33% | 93.27% |

### Detailed Classification Report (Best Model - Logistic Regression)
```text
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        10
Iris-versicolor       1.00      1.00      1.00        10
 Iris-virginica       1.00      1.00      1.00        10

       accuracy                           1.00        30
      macro avg       1.00      1.00      1.00        30
   weighted avg       1.00      1.00      1.00        30
```

### Visualizations Saved
All plots generated during the pipeline can be found in `reports/figures/`:
1. `pairplot.png`: Pairwise relationship of the features colored by species.
2. `correlation_heatmap.png`: Pearson correlation matrix of the features.
3. `confusion_matrix.png`: Prediction accuracy heatmap of the best model.
4. `feature_importance.png`: Feature weight/coefficient contribution plot for the best model.

