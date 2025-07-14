# Heart-Disease-Classification-using-Ensemble-Learning

# ❤️ Heart Disease Prediction using Ensemble Learning and Soft Voting

A machine learning project that predicts the likelihood of heart disease using supervised learning algorithms and ensemble techniques.

## 🚀 Project Overview

Heart disease is one of the leading causes of death worldwide. Early prediction can save lives by enabling timely medical intervention. This project builds a robust machine learning model that classifies whether a patient is likely to have heart disease based on clinical features like age, cholesterol levels, resting blood pressure, and other medical indicators.

## 🩺 Problem Statement

Develop an accurate predictive model for heart disease detection using patient medical data to:
- Enable early diagnosis and intervention
- Reduce mortality rates through timely treatment
- Assist healthcare professionals in decision-making
- Provide a reliable screening tool for at-risk patients

## 📊 Dataset

The dataset contains 13 clinical features for heart disease prediction:

| Feature | Description |
|---------|-------------|
| `age` | Age of the patient |
| `sex` | Gender (0 = female, 1 = male) |
| `cp` | Chest pain type (0-3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| `restecg` | Resting electrocardiographic results (0-2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes, 0 = no) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment (0-2) |
| `ca` | Number of major vessels colored by fluoroscopy (0-3) |
| `thal` | Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect) |
| `target` | **Target variable**: 1 = heart disease, 0 = no heart disease |

## ⚙️ Project Workflow

### 1. Data Preprocessing
- **Data Loading**: Imported dataset using pandas
- **Missing Value Handling**: Checked and handled null values
- **Feature Encoding**: Applied label encoding for categorical variables
- **Feature Scaling**: Standardized numerical features using StandardScaler
- **Data Splitting**: Split into training and testing sets (80/20 split)

### 2. Feature Selection Analysis

Performed comprehensive feature selection using multiple techniques to identify the most important features:

#### Feature Selection Methods Applied:
1. **Pearson Correlation** - Linear correlation with target variable
2. **Chi-Square Test** - Statistical significance for categorical features
3. **Recursive Feature Elimination (RFE)** - Backward elimination approach
4. **Logistic Regression** - Embedded feature selection
5. **Random Forest** - Tree-based feature importance
6. **LightGBM** - Gradient boosting feature importance

#### Feature Ranking Implementation:
```python
import numpy as np
import pandas as pd

# Create comprehensive feature selection dataframe
feature_selection_df = pd.DataFrame({
    'Feature': X.columns,
    'Pearson': cor_support,
    'Chi-2': chi_support,
    'RFE': rfe_support,
    'Logistics': embeded_lr_support,
    'Random Forest': embeded_rf_support,
    'LightGBM': embeded_lgb_support
})

# Count selection frequency across all methods
feature_selection_df['Total'] = feature_selection_df.drop(columns=['Feature']).sum(axis=1)

# Rank features by total selection count
feature_selection_df = feature_selection_df.sort_values(['Total', 'Feature'], ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df) + 1)

# Display top features
top_features = feature_selection_df.head(num_feats)
```

This analysis helped identify the most consistently important features across different selection methods, ensuring robust feature selection for model training.

### 3. Model Training & Evaluation

Trained and evaluated 10 different machine learning algorithms:

| Algorithm | Variable Name | Type |
|-----------|---------------|------|
| Random Forest | `y_pred_rfe` | Ensemble |
| Multi-layer Perceptron (MLP) | `y_pred_mlp` | Neural Network |
| K-Nearest Neighbors | `y_pred_knn` | Instance-based |
| Extra Trees Classifier | `y_pred_et_100` | Ensemble |
| XGBoost Classifier | `y_pred_xgb` | Gradient Boosting |
| Support Vector Classifier | `y_pred_svc` | SVM |
| Stochastic Gradient Descent | `y_pred_sgd` | Linear |
| AdaBoost Classifier | `y_pred_ada` | Boosting |
| Decision Tree (CART) | `y_pred_decc` | Tree-based |
| Gradient Boosting Machine | `y_pred_gbm` | Gradient Boosting |

### 4. Soft Voting Ensemble

Selected the top 5 performing models for ensemble learning:

#### 🧠 What is Soft Voting?
Soft voting combines predictions by averaging the predicted probabilities from multiple models rather than just counting votes. This approach:
- Considers the confidence level of each model's prediction
- Provides more stable and accurate results
- Reduces overfitting and improves generalization

#### Selected Models for Ensemble:
1. **Random Forest Classifier** (Weight: 4)
2. **Decision Tree Classifier** (Weight: 1)
3. **XGBoost Classifier** (Weight: 3)
4. **Extra Trees Classifier** (Weight: 5)
5. **Gradient Boosting Classifier** (Weight: 2)

```python
from sklearn.ensemble import VotingClassifier

eclf1 = VotingClassifier(
    estimators=[
        ('rfe', clf1),      # Random Forest
        ('decc', clf2),     # Decision Tree
        ('xgb', clf3),      # XGBoost
        ('ET', clf4),       # Extra Trees
        ('gb', clf5),       # Gradient Boosting
    ],
    voting='soft',
    weights=[4, 1, 3, 5, 2]
)

eclf1.fit(X_train, y_train)
y_pred_sv = eclf1.predict(X_test)
```

## 📈 Evaluation Metrics

Each model was evaluated using:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Ratio of true positives to predicted positives
- **Recall**: Ratio of true positives to actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of prediction performance

## 🏆 Results

The soft voting ensemble achieved the best overall performance, outperforming all individual models in terms of:
- Balanced accuracy
- F1-score
- Reduced overfitting
- Better generalization

## 🛠️ Technologies Used

- **Python 3.10**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms and tools
- **XGBoost** - Gradient boosting framework
- **LightGBM** - Gradient boosting framework for feature selection
- **Matplotlib/Seaborn** - Data visualization

