# Cardiovascular-Disease-Risk-Prediction


Aim
Develop a machine learning pipeline that predicts the risk of cardiovascular disease using clinical data from two datasets — the Cleveland Heart Disease dataset and the Pima Indians Diabetes dataset — by applying and comparing multiple regression and classification models.

Problem Statement
Cardiovascular disease (CVD) remains one of the leading causes of mortality worldwide. Early and accurate identification of individuals at risk can significantly reduce fatality rates by enabling timely medical intervention. Traditional diagnostic approaches depend heavily on specialist judgment and may be inconsistent across different clinical settings.

This mini project addresses the challenge by building a data-driven risk assessment system. Using structured clinical features such as age, blood pressure, cholesterol levels, and heart rate, the system learns patterns associated with disease presence. The project applies both regression models (to quantify a continuous risk score) and classification models (to provide a definitive Yes/No diagnosis), enabling a comprehensive evaluation of cardiovascular risk. 
Datasets Used
Dataset	Source	Records	Features	Target Variable
Cleveland Heart Disease	UCI ML Repository (processed_cleveland.csv)	303	13 Clinical	Heart disease present (0/1)
Pima Indians Diabetes	Kaggle (diabetes.csv)	768	8 Clinical	Diabetes outcome (0/1)


Target Variable
Heart Disease Dataset: target (binary) — 0 = No disease, 1 = Disease present (derived from the 'num' column; any value > 0 is mapped to 1)
Diabetes Dataset: Outcome (binary) — 0 = Non-diabetic, 1 = Diabetic

 
Model Pipeline

The following pipeline is applied uniformly to both datasets for a consistent, reproducible ML workflow:

1	Data Loading & Understanding
Both datasets are loaded using pd.read_csv().
For the Cleveland dataset, '?' values (missing markers) are replaced with NaN and all columns are cast to numeric.
The target column 'num' is binarised: values > 0 → 1 (disease), 0 → 0 (no disease); the original column is then dropped.
df.info() and df.describe() are used to understand structure, data types, and basic statistics.

2	Data Preprocessing
Zero-replacement for physiologically impossible zeros: In the Diabetes dataset, columns such as Glucose, BloodPressure, SkinThickness, Insulin, and BMI cannot be zero; these are replaced with NaN before imputation.
Missing value imputation: SimpleImputer (strategy='median') fills all remaining NaN values with the column median, preserving distribution robustness against outliers.
Feature–Target split: X (input features) and y (target variable) are separated.
Train–Test split: Data is divided 80:20 using train_test_split() with stratify=y to maintain class balance.
Feature scaling: StandardScaler is applied — fit on training data, then transform applied to both train and test sets to prevent data leakage.

3	Model Selection & Training
Regression Models (for continuous risk score): Linear Regression, Ridge Regression (alpha=1), Lasso Regression (alpha=0.1).
Classification Models (for binary disease prediction): Logistic Regression with L1 penalty, Logistic Regression with L2 penalty, Decision Tree Classifier (max_depth=5), Random Forest Classifier.
All models are trained on the scaled training set using model.fit().
Both regression and classification experiments are run independently on both datasets.

4	Prediction
Trained models generate predictions on the held-out test set using model.predict().
For the Random Forest classifier, model.predict_proba() is also used to obtain class probabilities.
A sample patient record is constructed manually and passed through the trained scaler before prediction, demonstrating real-world inference.

5	Model Evaluation
Regression metrics: RMSE (Root Mean Squared Error) — lower is better; R² Score — closer to 1.0 is better.
Classification metrics: Accuracy, Precision, Recall, F1-Score — all computed using sklearn.metrics.
Confusion Matrix: Visualised using seaborn heatmap to show true/false positive and negative counts.

6	Visualisation & Comparative Analysis
RMSE Comparison (bar chart): Compares Linear, Ridge, and Lasso regression performance side by side.
Classification Metrics (grouped bar chart): Plots Accuracy and F1-Score for all four classifiers.
Feature Importance (horizontal bar chart): Displays the top contributing features from the Random Forest model.
Confusion Matrix Heatmap: Annotated matrix showing true/false positive/negative classifications.
User vs. Dataset Comparison (grouped bar chart): Compares individual patient values against dataset averages for each feature.
Risk Score Bar: Visualises the regression-predicted risk score with threshold markers (0.3 and 0.6).
Disease Probability Bar: Shows predicted class probabilities (No Disease / Disease) from the classifier.
