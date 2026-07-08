<h1 align="center">🌸 Iris Flower Classification</h1>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter" />
</div>

<br/>

<p align="center">
  <strong>A professional Machine Learning pipeline to accurately classify Iris flowers based on morphological measurements.</strong>
</p>

## 📖 About the Iris Dataset

The [Iris flower dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) is one of the most famous datasets in machine learning and statistics, introduced by the British statistician and biologist Ronald Fisher in 1936. 

The dataset consists of **150 samples** from three species of Iris flowers:
- 🌸 **Setosa** (`Iris-setosa`)
- 🌼 **Versicolor** (`Iris-versicolor`)
- 🌺 **Virginica** (`Iris-virginica`)

For each sample, **4 quantitative features** were measured (in centimeters):
1. Sepal Length
2. Sepal Width
3. Petal Length
4. Petal Width

## 🤖 Machine Learning Approach

This project implements a robust ML pipeline emphasizing modularity, hyperparameter tuning, and comprehensive evaluation. The workflow includes:

1. **Data Preprocessing**: Handling missing values, label encoding species, and scaling features using `StandardScaler`.
2. **Exploratory Data Analysis (EDA)**: Visualizing distributions and pairwise correlations among the measurements.
3. **Model Training & Tuning**: Applying Grid Search with 5-fold Cross-Validation across multiple algorithms to find the optimal hyperparameters.
4. **Evaluation**: Generating detailed classification metrics and confusion matrices.
5. **Inference Pipeline**: Saving the best model pipeline (scaler + classifier) to accurately predict on unseen data.

## 📊 Feature Analysis

Based on the Exploratory Data Analysis, petal dimensions (Length and Width) are the most crucial features for accurately distinguishing the species:
- **Setosa** is highly linearly separable from the other two species, exhibiting significantly smaller petal lengths and widths.
- **Versicolor & Virginica** have some overlap but can be distinguished as Virginica generally has larger petals and sepals.
- **Sepal dimensions** provide additional context but are less discriminative on their own compared to petal measurements.

## 🏆 Model Performance Comparison

Multiple algorithms were evaluated to find the most accurate classifier for the dataset:

| Algorithm | Best Parameters | Test Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **🏆 Logistic Regression** | `{'C': 10.0, 'solver': 'lbfgs'}` | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| **Random Forest** | `{'max_depth': 3, 'min_samples_split': 2, 'n_estimators': 50}` | 96.67% | 96.97% | 96.67% | 96.66% |
| **Support Vector Machine** | `{'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'}` | 93.33% | 93.33% | 93.33% | 93.33% |
| **K-Nearest Neighbors** | `{'metric': 'euclidean', 'n_neighbors': 5}` | 93.33% | 94.44% | 93.33% | 93.27% |

*Logistic Regression proved to be the most effective algorithm in our pipeline, correctly identifying 100% of the test samples.*

## 📈 Visualizations Generated

The training pipeline automatically generates the following visual reports (found in `Task_1/reports/figures/`):

- 📉 **Pairplot**: Displays the pairwise relationships of all 4 features, colored by species class, demonstrating the high separability of *Iris-setosa*.
- 🌡️ **Correlation Heatmap**: Shows the Pearson correlation coefficient between all features, highlighting the strong positive correlation between petal length and petal width.
- 🎯 **Confusion Matrix**: A heatmap of the best model's predictions vs. actual labels, providing insight into any misclassifications.
- 📊 **Feature Importance**: Plots the weights or coefficients of the chosen algorithm to indicate which feature had the highest predictive power.

## 💻 Tech Stack

| Category | Technology |
| :--- | :--- |
| **Programming Language** | Python 3.8+ |
| **Data Manipulation** | Pandas, NumPy |
| **Machine Learning** | scikit-learn |
| **Visualizations** | Matplotlib, Seaborn |
| **Development Tool** | Jupyter Notebook, VS Code |

## 🚀 How to Run

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/deveshkumawat258/Iris_flower_Classification.git
cd Iris_flower_Classification/Task_1
pip install -r requirements.txt
```

### 2. Training the Model
Run the automated pipeline to process data, train all models, generate plots, and save the best model:
```bash
python src/train.py
```

### 3. Interactive Predictions
Predict the species of a new flower via the interactive CLI:
```bash
python src/predict.py
```
*(You will be prompted to enter the Sepal and Petal measurements).*

## 👨‍💻 Author

**Devesh Kumawat**
- GitHub: [deveshkumawat258](https://github.com/deveshkumawat258)
- Repository: [Iris_flower_Classification](https://github.com/deveshkumawat258/Iris_flower_Classification)

---
<p align="center"><i>If you found this project helpful, please consider giving it a ⭐ on GitHub!</i></p>
