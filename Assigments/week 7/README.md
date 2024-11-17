# Weather Forecasting Project (Rain Prediction)

This project focuses on predicting whether it will rain or not using historical weather data. The pipeline includes data exploration, preprocessing, model training, hyperparameter tuning, and evaluation.

## Project Overview

The goal is to classify weather conditions into two categories: **Rain** or **No Rain** based on historical data. The process involves:

1. **Data Loading & Exploration**  
2. **Exploratory Data Analysis (EDA)**  
3. **Data Preprocessing**  
4. **Modeling & Hyperparameter Tuning**  
5. **Model Evaluation**

## Workflow

### 1. Data Loading and Exploration
- The dataset contains various weather features and a binary target variable indicating whether it rained (`1`) or not (`0`).
- Basic checks were performed to understand data structure, missing values, and data types.

### 2. Exploratory Data Analysis (EDA)
EDA steps include:
- **Summary Statistics**: Analyzed distributions of key weather features.
- **Correlation Heatmap**: Checked relationships between predictors.
- **Class Distribution**: Visualized the balance between rain and no-rain cases.

### 3. Data Preprocessing
- **Handling Missing Values**: Removed rows with missing values to ensure data integrity.
- **Feature Encoding**: Transformed categorical variables into numerical values.
- **Data Splitting**: Split the data into training and testing sets.

### 4. Modeling & Hyperparameter Tuning

#### **1. Logistic Regression**
- **Cross-Validation**: 5-Fold Cross-Validation for evaluation.
- **Evaluation Metric**: ROC AUC Score.
- **Average ROC AUC Score**: 0.964

#### **2. Decision Tree Classifier**
- **Hyperparameters Tuned**:
  - `max_depth`
  - `min_samples_split`
- **Best ROC AUC Score**: 0.98

#### **3. Random Forest Classifier**
- **Hyperparameters Tuned**:
  - `n_estimators`
  - `max_depth`
  - `min_samples_split`
- **Best ROC AUC Score**: 0.988

#### **4. XGBoost Classifier**
- **Hyperparameters Tuned**:
  - `n_estimators`
  - `learning_rate`
  - `max_depth`
- **Best ROC AUC Score**: 0.994

### 5. Model Evaluation
- **Primary Metric**: ROC AUC Score.
- **Best Model**: XGBoost Classifier with a ROC AUC Score of **0.994**.
- The evaluation demonstrated that the XGBoost model outperformed others, making it the most suitable for predicting rainfall.

## Results and Conclusion
- **XGBoost Classifier** emerged as the best-performing model, accurately classifying rain and no-rain instances.
- This project illustrates the effectiveness of tree-based methods combined with hyperparameter tuning for binary classification problems.

## How to Run
1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
