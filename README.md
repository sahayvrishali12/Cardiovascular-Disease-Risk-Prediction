# Cardiovascular Disease Risk Prediction

## Aim

Develop a machine learning pipeline that predicts the risk of cardiovascular disease using clinical data from two datasets: the Cleveland Heart Disease dataset and the Pima Indians Diabetes dataset. The project applies and compares multiple regression and classification models.

---

## Problem Statement

Cardiovascular disease (CVD) is one of the leading causes of mortality worldwide. Early and accurate identification of individuals at risk can significantly reduce fatality rates by enabling timely medical intervention.

Traditional diagnostic approaches depend heavily on specialist judgment and may be inconsistent across different clinical settings. This project addresses the problem by building a data-driven risk assessment system using structured clinical features such as age, blood pressure, cholesterol levels, and heart rate.

The system uses:

* Regression models to generate a continuous risk score
* Classification models to provide a binary diagnosis (Yes/No)

---

## Datasets Used

| Dataset                 | Source            | Records | Features | Target Variable        |
| ----------------------- | ----------------- | ------- | -------- | ---------------------- |
| Cleveland Heart Disease | UCI ML Repository | 303     | 13       | Heart disease (0/1)    |
| Pima Indians Diabetes   | Kaggle            | 768     | 8        | Diabetes outcome (0/1) |

---

## Target Variables

* Cleveland Dataset

  * `target`: 0 = No disease, 1 = Disease present
  * Derived from `num` column (num > 0 → 1)

* Diabetes Dataset

  * `Outcome`: 0 = Non-diabetic, 1 = Diabetic

---

## Data Statistics

### Cleveland Heart Disease Dataset

| Feature                | Mean | Std Dev | Min | Max |
| ---------------------- | ---- | ------- | --- | --- |
| Age                    | 54.4 | 9.0     | 29  | 77  |
| Resting Blood Pressure | 131  | 17      | 94  | 200 |
| Cholesterol            | 246  | 51      | 126 | 564 |
| Max Heart Rate         | 150  | 22      | 71  | 202 |

---

### Pima Indians Diabetes Dataset

| Feature        | Mean | Std Dev | Min | Max |
| -------------- | ---- | ------- | --- | --- |
| Glucose        | 120  | 32      | 0   | 199 |
| Blood Pressure | 69   | 19      | 0   | 122 |
| BMI            | 32   | 7.9     | 0   | 67  |
| Insulin        | 80   | 115     | 0   | 846 |

---

### Missing Values Handling

| Dataset   | Columns Affected                         | Handling Method     |
| --------- | ---------------------------------------- | ------------------- |
| Cleveland | '?' values                               | Converted to NaN    |
| Diabetes  | Glucose, BP, SkinThickness, Insulin, BMI | 0 replaced with NaN |
| Both      | All columns                              | Median imputation   |

---

## Machine Learning Pipeline

### 1. Data Loading and Understanding

* Data loaded using `pd.read_csv()`
* Missing values handled
* Target column binarised for Cleveland dataset
* `df.info()` and `df.describe()` used for analysis

---

### 2. Data Preprocessing

* Invalid zero values replaced with NaN (Diabetes dataset)
* Missing values filled using median imputation
* Feature-target split into X and y
* Train-test split (80:20) with stratification
* Feature scaling using StandardScaler

---

### 3. Models Used

#### Regression Models

* Linear Regression
* Ridge Regression (alpha = 1)
* Lasso Regression (alpha = 0.1)

#### Classification Models

* Logistic Regression (L1 regularization)
* Logistic Regression (L2 regularization)
* Decision Tree (max depth = 5)
* Random Forest

---

## Model Evaluation

### Regression Results

| Model             | RMSE (Cleveland) | RMSE (Diabetes) | R² Score |
| ----------------- | ---------------- | --------------- | -------- |
| Linear Regression | 0.42             | 0.45            | 0.30     |
| Ridge Regression  | 0.41             | 0.44            | 0.32     |
| Lasso Regression  | 0.43             | 0.46            | 0.28     |

---

### Classification Results

| Model                    | Accuracy | Precision | Recall | F1 Score |
| ------------------------ | -------- | --------- | ------ | -------- |
| Logistic Regression (L1) | 0.82     | 0.80      | 0.78   | 0.79     |
| Logistic Regression (L2) | 0.83     | 0.81      | 0.80   | 0.80     |
| Decision Tree            | 0.78     | 0.75      | 0.76   | 0.75     |
| Random Forest            | 0.86     | 0.84      | 0.85   | 0.84     |

---

## Prediction

* Predictions generated using `model.predict()`
* Probabilities obtained using `predict_proba()` (Random Forest)
* Supports custom input for real-world inference after scaling

---

## Visualizations

* RMSE comparison (regression models)
* Accuracy and F1-score comparison (classification models)
* Feature importance (Random Forest)
* Confusion matrix heatmap
* User vs dataset feature comparison
* Risk score visualization with thresholds (0.3 and 0.6)
* Disease probability bar chart
