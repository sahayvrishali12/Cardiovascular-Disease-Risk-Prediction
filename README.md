# Applied Machine Learning — Laboratory File

**Submitted To:** Dr. Sahinur Rahman Laskar, Assistant Professor, SoCS, UPES
**Submitted By:** Vrishali Sahay | SAP ID: 590011701 | Batch: 19
**School of Computer Science, UPES, Dehradun**
**B.TECH. — IV Semester | Jan – May 2026**

---

## Index

| S.No | Experiment | Date |
|------|------------|------|
| 1 | Data Preprocessing | 27-01-26 |
| 2 | Predicting House Price using Linear Regression | 03-02-26 |
| 3 | Predicting Stock Price using Regression | 10-02-26 |
| 4 | Predicting Customer Churn Rate using Logistic Regression | 10-02-26 |
| 5 | Spam Detection (Classification and Comparative Analysis) | 24-02-26 |
| 6 | Credit Risk Assessment and Comparative Analysis | 24-02-26 |
| 7 | Anomaly Detection and Comparative Analysis | 24-02-26 |
| 8 | Student Performance Level Analysis | 17-02-26 |
| 9 | Physiological Signal Classification | 24-02-26 |
| 10 | Iris Classification | 24-02-26 |
| 11 | Diabetes Prediction using Ensemble Learning | 02-03-2026 |
| 12 | L1 & L2 Regularization on Melbourne Housing Dataset | 02-03-2026 |
| 13 | Cardiovascular Disease Risk Prediction | 09-03-2026 |

---

## Experiment 1 — Data Preprocessing

### Aim
Text pre-processing in IPL dataset using Python libraries.

### Problem Statement

Raw datasets derived from real-world events like IPL cricket matches often contain inconsistencies, missing values, duplicate entries, and unstructured text that cannot be directly used for analysis or model training. This experiment focuses on cleaning and transforming the IPL dataset into a structured, analysis-ready format using standard Python libraries.

### Dataset

The IPL (Indian Premier League) dataset contains match-level and player-level information including team names, venues, player names, dismissal types, and match results. These fields are primarily textual and require preprocessing before any downstream analysis or machine learning can be applied.

**Libraries Used:**
- **Pandas:** Data loading, manipulation, and cleaning
- **NumPy:** Numerical operations and handling of missing values
- **NLTK / re:** Text cleaning and pattern-based preprocessing
- **Matplotlib / Seaborn:** Visualizing distributions before and after preprocessing

### Pipeline

#### 1. Data Loading and Exploration
- Dataset loaded using `pd.read_csv()`
- Initial exploration using `df.head()`, `df.info()`, and `df.describe()`
- Shape of the dataset checked to understand the number of records and features
- Column data types inspected to identify text vs. numerical fields

#### 2. Handling Missing Values
- Missing entries identified using `df.isnull().sum()`
- Columns with high null percentages (e.g., `player_dismissed`, `dismissal_kind`) noted
- Null values in categorical columns filled with placeholder strings like `"none"` or `"unknown"`
- Rows with critical missing fields dropped using `df.dropna(subset=[...])`

#### 3. Removing Duplicates
- Duplicate rows detected using `df.duplicated().sum()`
- Duplicates removed using `df.drop_duplicates()` to ensure data integrity

#### 4. Text Cleaning
- Team names and venue names standardized (e.g., "Delhi Daredevils" → "Delhi Capitals")
- Player names trimmed of leading/trailing whitespace using `str.strip()`
- Special characters and extra spaces removed using `str.replace()` and regex (`re` module)
- All text fields converted to lowercase for uniformity using `str.lower()`

#### 5. Encoding Categorical Variables
- Categorical columns such as `batting_team`, `bowling_team`, `dismissal_kind` encoded using:
  - **Label Encoding** for ordinal or binary categories
  - **One-Hot Encoding** (`pd.get_dummies()`) for nominal multi-class categories

#### 6. Feature Engineering
- New features derived from existing columns (e.g., extracting `season year` from match date)
- Over number and ball number combined or separated as needed
- Runs per over calculated as an aggregate feature for match-level analysis

#### 7. Data Validation and Export
- Final cleaned dataset checked using `df.info()` and `df.describe()` to confirm no nulls remain
- Data types verified and corrected where necessary
- Cleaned dataset exported using `df.to_csv()` for use in subsequent experiments

### Preprocessing Summary

| Step | Operation | Method / Function Used |
|------|-----------|------------------------|
| Data Loading | Load raw IPL CSV | `pd.read_csv()` |
| Exploration | Inspect structure and types | `df.info()`, `df.describe()`, `df.head()` |
| Missing Values | Identify and fill nulls | `df.isnull().sum()`, `df.dropna()`, fill with `"none"` |
| Duplicates | Detect and remove duplicates | `df.duplicated()`, `df.drop_duplicates()` |
| Text Cleaning | Standardize names, strip whitespace | `str.strip()`, `str.lower()`, `str.replace()`, regex |
| Encoding | Convert categorical to numerical | `pd.get_dummies()`, Label Encoding |
| Feature Engineering | Derive new features | `shift()`, aggregation functions |
| Export | Save cleaned dataset | `df.to_csv()` |

### Key Takeaways
- Raw IPL data contained significant missing values in player and dismissal-related fields, which were handled using context-aware imputation strategies.
- Standardizing team names and removing inconsistencies was critical to ensure groupby and aggregation operations produced accurate results.
- One-hot encoding of categorical features ensured compatibility with machine learning algorithms in later experiments.
- The preprocessing pipeline reduced noise in the dataset and improved overall data quality, forming a reliable foundation for analysis and modelling.

---

## Experiment 2 — Predicting Housing Prices

### Aim
Develop a regression model to predict house prices based on features like location, size, and amenities.

### Problem Statement

The determination of real estate market value is a complex process influenced by a diverse interplay of structural, locational, and economic variables. Historically, property valuation has relied heavily on manual appraisals and the subjective expertise of agents, which can often be inconsistent, time-consuming, or influenced by human bias. As the real estate market grows in complexity, there is an increasing demand for objective, data-driven solutions that can provide rapid and accurate valuations without the variance found in manual estimations.

This project addresses these inefficiencies by seeking to automate the valuation process through the identification of intricate patterns within historical housing data. Given a dataset containing 120 detailed property records, the primary objective is to develop a robust regression model capable of transforming property attributes into precise market estimates. By leveraging machine learning, we aim to move toward a reliable, evidence-based framework that ensures transparency and mathematical consistency for both buyers and sellers.

**Input Features:**
- **Location:** The neighborhood type (Rural, Urban, Suburban)
- **Area_sqft:** The total floor area of the house in square feet
- **Bedrooms/Bathrooms:** The count of essential living spaces
- **Balcony/Parking:** Additional amenities available
- **Age_Years:** The age of the property (to account for depreciation or vintage value)
- **Furnished:** Status of the interior furnishing (Yes/No)

**Target Variable:**
- **Price:** The estimated market value of the house

### Model Pipeline

#### 1. Data Understanding
- **Target Variable (y):** The target is to calculate the price of the house — the numerical value the model learns to predict.
- **Input Features:**
  - **Numerical:** Area_sqft, Bedrooms, Bathrooms, Age_Years, Parking, and Balconies
  - **Categorical:** Location (Urban, Suburban, Rural) and Furnished (Yes, No)
- Data is loaded using `pd.read_csv()` and inspected with `df.info()` to verify each row represents an individual property record.

#### 2. Data Preprocessing
- **Handling Categorical Variables:** Text-based categories like "Urban" or "Suburban" are converted into numbers using One-Hot Encoding (`pd.get_dummies()`), creating binary columns like `Location_Urban` and `Furnished_Yes`, avoiding false ordinal relationships.
- **Feature and Target Split:** Data is separated into X (all input columns) and y (the Price column).

#### 3. Train–Test Split
Using `train_test_split()` to divide data into:
- **Training set (80%):** To learn the relationships
- **Test set (20%):** To evaluate performance

#### 4. Regression Model & Training
- **Model Selection:** Linear Regression is chosen to learn how each feature contributes to the final price.
- **Training:** Using `model.fit()`, the model calculates coefficients (weights) that minimize prediction error on the training data.

#### 5. Predictions
- The model is applied to unseen test data using `model.predict()` on `X_test`.
- Actual prices are plotted against predicted prices to visually confirm the model correctly follows market trends.

#### 6. Model Evaluation
Performance is assessed using Mean Squared Error (MSE) and R² via `sklearn.metrics`:
- **Lower MSE** indicates a better fit, as it penalizes large errors strongly.
- **R² Score** tells us how much of the price variance is explained by our input features.

#### 7. Model Saving
- The trained model is saved using the Pickle library (`pickle.dump()`) as a `.pkl` file.
- It can be loaded later using `pickle.load()` for instant predictions without retraining.

### Model Results

| Metric | Value |
|--------|-------|
| Model Used | Linear Regression |
| Dataset Size | 120 records |
| Train Size | 96 records (80%) |
| Test Size | 24 records (20%) |
| Evaluation Metric 1 | Mean Squared Error (MSE) — lower is better |
| Evaluation Metric 2 | R² Score — higher is better |
| Model Saved | Yes — `.pkl` via `pickle.dump()` |

### Key Takeaways
- The linear regression model successfully learned the relationship between housing features such as location, area, amenities, and age with the house price, producing stable and consistent predictions.
- The model achieved very low mean squared error and a high R² score, indicating strong predictive performance.
- One-hot encoding of categorical variables ensured all input features contributed fairly to the regression model.
- Saving both the trained model and the scaler ensures the model can be reliably reused for future predictions on new housing data.

---

## Experiment 3 — Stock Price Prediction

### Aim
Develop a time series prediction model to forecast stock prices.

### Problem Overview

The stock market is a dynamic environment where price determination is driven by a complex interplay of historical trends, market sentiment, and macroeconomic shifts. Traditional manual forecasting methods often struggle to maintain consistency, as human analysts can be biased or overwhelmed by the sheer volume of sequential data points. This project aims to address these challenges by developing an automated valuation system that identifies hidden temporal patterns within historical stock records.

By analyzing a dataset of stock records, the task is to build a robust regression model that can predict future market values based on past performance. Moving away from subjective estimates, this project implements a data-driven framework where the model learns from historical "lags" to forecast future outcomes, providing a consistent and transparent tool for investors and analysts.

### Model Pipeline

#### 1. Understand the Dataset
- **Temporal Data:** Each row represents a specific point in time, tracking the movement of various stocks over a sequential period.
- **Target Variable (y):** Price — The continuous value to forecast for the current day.
- **Input Features (X):**
  - **Numerical (Lags):** Prices from the previous 5 days (lag1 to lag5)
  - **Categorical:** Stock ticker (e.g., AAPL) identifying the specific company being forecast
- **Excluded Features:** Same-day Open, High, Low, Close, and Volume are intentionally excluded to prevent data leakage.

#### 2. Data Preprocessing
- **Chronological Alignment & Encoding:**
  - The Date column is converted using `pd.to_datetime()` and records sorted by time.
  - One-Hot Encoding (`pd.get_dummies()`) converts stock names into binary numbers, allowing the model to distinguish between stocks without false ordinal assumptions.
- **Lag Feature Engineering:**
  - 5 new columns are created using the `shift()` operation, representing prices from the last 5 days.
- **Feature-Target Split:**
  - X contains lag features and stock dummies; Y contains the target Price.

#### 3. Train–Test Split
Because this is time-series data, records are not shuffled:
- **Training set:** First 80% of the historical timeline
- **Test set:** Final 20% reserved to evaluate true forecasting accuracy

#### 4. Choose and Train the Model
- **Model:** Linear Regression identifies the mathematical trend line connecting past prices to future values.
- The `.fit()` function calculates weights (coefficients) for each lag, determining the contribution of each past day's price to minimize prediction error.

#### 5. Predictions
- The trained model is applied to the unseen test set using `.predict()`.
- Actual prices are plotted against predicted prices to visually confirm the model correctly follows market trends.

#### 6. Model Evaluation

| Metric | Value |
|--------|-------|
| MAE | 1.27668 |
| R² | 1.00 |

#### 7. Model Saving
- The Pickle library (`pickle.dump()`) is used to serialize the trained Linear Regression model into a `.pkl` file.
- This allows for instant loading and real-time forecasting in future sessions without retraining on old data.

### Model Results

| Setting | Value |
|---------|-------|
| Model Used | Linear Regression (Lag-based Time Series) |
| Lag Features | lag1, lag2, lag3, lag4, lag5 |
| Train Split | 80% (chronological — no shuffle) |
| Test Split | 20% (chronological) |
| MAE | 1.27668 |
| R² Score | 1.00 |
| Model Saved | Yes — `.pkl` via `pickle.dump()` |

---

## Experiment 4 — Customer Churn Prediction

### Aim
Customer Churn Prediction: Develop a model to predict customer churn in a subscription-based business.

### Problem Statement

Customer retention is a major challenge for service-based businesses, as acquisition costs often exceed retention costs. Manual tracking typically fails to identify the subtle behavioral patterns that signal a churn event. This project aims to automate the identification of at-risk customers by analyzing a dataset of 200 records including historical usage and contract data.

Unlike prior regression tasks, this is a classification problem focused on predicting a discrete category: Churn (1) or No Churn (0). Implementing this data-driven framework allows the business to transition from reactive to proactive retention strategies. By targeting interventions toward users most likely to depart, the business can optimize marketing resources and stabilize long-term revenue.

### Project Pipeline

#### 1. Data Understanding
The dataset consists of 200 customer records with 11 attributes.
- **Target Variable (y):** Churn
- **Numerical Features:** Age, Tenure_Months, MonthlyCharges, TotalCharges, SupportTickets
- **Categorical Features:** Gender, SubscriptionType, PaymentMethod, ContractType

#### 2. Data Preprocessing
- **Feature Selection:** The `CustomerID` column is dropped using `df.drop()` as it is a unique identifier with no predictive value.
- **Cleaning:** Rows with missing values are removed via `df.dropna()`.
- **Categorical Encoding:** Text-based features are converted into numerical binary vectors using One-Hot Encoding (`pd.get_dummies()`).

#### 3. Feature-Target & Train-Test Split
- **Ratio:** 80% of the data is used for training, and 20% is reserved for testing.
- **Test Sample:** 40 unseen records are used to validate the model's accuracy.

#### 4. Choose and Train the Model
Logistic Regression is selected for its simplicity and interpretability in binary classification tasks.
- The model is initialized with `max_iter=1000` and trained using the `.fit()` method.

#### 5. Make Predictions
- The `.predict()` function is used on `X_test`, producing an array of binary predictions (0 or 1), assigning a churn status to each customer in the test group.

#### 6. Evaluate the Model
- **Accuracy:** 75% (The model correctly identified 30 out of 40 customers)
- **Class 0 (Stayed):** Precision and Recall are high at approximately 85%
- **Class 1 (Churned):** Precision and Recall are low at ~29%, indicating the model struggles to catch the minority churn class

#### 7. Model Saving
- The Pickle library (`pickle.dump()`) is used to serialize the model object into a permanent `.pkl` file.
- The saved file allows deployment in a business dashboard for future use without retraining.

### Model Results

| Metric | Class 0 (No Churn) | Class 1 (Churn) | Overall |
|--------|--------------------|-----------------|---------|
| Precision | ~0.85 | ~0.29 | — |
| Recall | ~0.85 | ~0.29 | — |
| F1-Score | ~0.85 | ~0.29 | — |
| Accuracy | — | — | 0.75 (30/40 correct) |

| Setting | Value |
|---------|-------|
| Model | Logistic Regression |
| max_iter | 1000 |
| Dataset Size | 200 records |
| Train / Test Split | 80% / 20% (160 / 40 records) |
| Model Saved | Yes — `.pkl` via `pickle.dump()` |

---

## Experiment 5 — Spam Detection

### Aim
Spam Email Detection: Build a spam email filter using text classification algorithms and perform comparative analysis. (Naive Bayes, Logistic Regression, Support Vector Machine, Decision Tree, Random Forest)

### Problem Statement

Spam detection is essential for maintaining email security and user productivity, as malicious or unsolicited emails can lead to security breaches or cluttered inboxes. Manual filtering is impossible given the volume of communication, necessitating automated classification models. This project aims to build a robust spam filter by performing a comparative analysis of five text classification algorithms: Naïve Bayes, Logistic Regression, Support Vector Machine (SVM), Decision Tree, and Random Forest. By evaluating these models, the goal is to identify which algorithm most accurately distinguishes between "Spam" and "Ham" (legitimate) emails.

### Pipeline

#### 1. Problem Definition
The primary goal is to develop a robust automated spam filter using advanced text classification techniques.
- **Classification Task:** The model must determine if an incoming email is Spam (1) or Ham (0).
- **Objective:** Perform a detailed comparative analysis across five distinct algorithms — Naïve Bayes, Logistic Regression, SVM, Decision Tree, and Random Forest — to identify the most reliable architecture for real-world deployment.

#### 2. Data Understanding
- **Dataset Content:** A collection of thousands of email text bodies labeled as spam or legitimate (ham).
- **Target (y):** Binary labels where 1 represents "Spam" and 0 represents legitimate "Ham."
- **Input Features (X):** The raw text content of emails, including subject lines and message bodies.

#### 3. Data Preprocessing
Since machine learning models cannot interpret raw text, the data is put through a rigorous cleaning and transformation process:
- **Text Cleaning:** Stripping out punctuation, special characters, and numbers to reduce noise.
- **Stop-word Removal:** Eliminating frequently used words (e.g., "and", "the", "is") that carry little predictive value.
- **Vectorization:** Using TF-IDF (Term Frequency-Inverse Document Frequency) to transform words into numerical vectors.

#### 4. Train-Test Split
- **Data Allocation:** Cleaned dataset split into Training set (80%) and Test set (20%).
- **Purpose:** This division prevents overfitting and allows measurement of real-world filter performance.

#### 5. Model Selection and Training
Five different classification models are initialized and trained on the same training data for a fair comparison:
1. **Naïve Bayes:** A probabilistic classifier based on Bayes' Theorem, highly efficient for high-dimensional text data.
2. **Logistic Regression:** A linear model that estimates the probability of an email belonging to the spam class.
3. **Support Vector Machine (SVM):** Finds the optimal hyperplane maximizing the distance between spam and ham clusters.
4. **Decision Tree:** Uses a tree-like structure of word-based logical decisions to categorize each message.
5. **Random Forest:** An ensemble method that constructs multiple decision trees and outputs the majority class.

#### 6. Predictions
- The `.predict()` function is called for each algorithm, generating five separate arrays of binary predictions.
- Predictions are compared against ground-truth labels to pinpoint errors such as False Positives (blocking a real email) or False Negatives (letting spam through).

#### 7. Model Comparison and Evaluation

| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| Naive Bayes | 0.976 | 1.0 | 0.825 |
| Logistic Regression | 0.967 | 1.0 | 0.758 |
| **SVM** | **0.984** | **1.0** | **0.880** |
| Decision Tree | 0.968 | 0.9 | 0.859 |
| Random Forest | 0.979 | 1.0 | 0.830 |

- **Accuracy:** SVM achieved the best accuracy (98.4%).
- **Precision:** Logistic Regression, Naive Bayes, SVM, and Random Forest all scored 1.0 — zero false positives.
- **Recall:** SVM achieved the highest recall (88%), meaning it catches the most actual spam.

**Overall, SVM is the best model** among all as it scored the highest accuracy and recall as well as perfect precision.

#### 8. Model Saving
- All trained models are saved using the Pickle library (`pickle.dump()`) serialized into permanent `.pkl` files.
- These files allow immediate deployment in an email system for real-time filtering without retraining.

---

## Experiment 6 — Credit Risk Assessment

### Aim
Build a credit scoring model to assess the creditworthiness of applicants using historical financial data and perform comparative analysis (Logistic Regression, Random Forest, XGBoost).

### Problem Definition

Automated credit scoring is vital for financial institutions to mitigate potential losses and optimize lending decisions. Because the volume of applications makes manual review inefficient, we require robust classification models to predict applicant creditworthiness. This project aims to build a reliable credit filter by performing a comparative analysis of three algorithms: Logistic Regression, Random Forest, and Gradient Boosting. By evaluating these models, we seek to identify the architecture that most accurately categorizes "High," "Medium," and "Low" risk applicants, ensuring proactive risk management and improved financial stability.

### Pipeline

#### 1. Problem Definition
- **Classification Task:** Determine the creditworthiness of an applicant, categorizing them into risk levels: High, Medium, or Low.
- **Objective:** Perform a detailed comparative analysis across Logistic Regression, Random Forest, and Gradient Boosting to identify the most reliable architecture for financial deployment.

#### 2. Data Understanding
- **Dataset Content:** A collection of 500 samples of applicant profiles including historical financial health indicators.
- **Input Features (X):** Age, annual income, employment years, credit score, loan amount, and late payments.
- **Target (y):** Categorical labels representing "Credit Risk" (High, Medium, or Low).

#### 3. Data Preprocessing
- **Feature Selection:** Drop non-predictive identifiers such as `Applicant_ID`.
- **Label Encoding:** Utilize a `LabelEncoder` to transform categorical risk labels into numerical vectors.
- **Standardization:** Apply a `StandardScaler` to normalize numerical features like income and loan amounts, preventing large values from dominating the model logic.

#### 4. Train-Test Split
- **Data Allocation:** Training set (80%) to build models and Test set (20%) to validate.
- **Purpose:** Prevents overfitting and allows measurement of real-world performance in a live financial environment.

#### 5. Model Selection and Training
Three different classification models are initialized and trained on the same data for a fair comparison:
- **Logistic Regression:** A baseline linear model to estimate risk probabilities based on weighted input features.
- **Random Forest:** An ensemble method that constructs multiple decision trees, improving stability and capturing non-linear relationships.
- **Gradient Boosting:** Corrects errors of previous trees sequentially to achieve high-fidelity predictions.

#### 6. Predictions
- **Execution:** Each model classifies the 100 unseen samples in the test set.
- **Output Generation:** Binary and multi-class prediction arrays are generated for each algorithm.

#### 7. Model Evaluation and Comparison

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Logistic Regression | 0.85 | 0.81 | 0.76 | 0.79 |
| Random Forest | 0.98 | 0.98 | 0.87 | 0.91 |
| **Gradient Boosting** | **1.00** | **1.00** | **1.00** | **1.00** |

**Gradient Boosting is the best model** among all tested as it scored the highest across all critical financial metrics and showed perfect classification capability on the test set.

#### 8. Model Saving
- All trained models, along with the scaler and label encoder, are saved using the Pickle library into permanent `.pkl` files.
- These files allow for immediate deployment in a banking system without the need for retraining.

---

## Experiment 7 — Anomaly Detection

### Aim
Implement an anomaly detection system for detecting outliers in data (e.g., fraud detection) and perform comparative analysis. (Isolation Forest, Local Outlier Factor, One-Class SVM)

### Problem

Anomaly detection is critical for identifying rare items, events, or observations which raise suspicions by differing significantly from the majority of the data. Because the volume of transactions makes manual inspection for fraud or errors impossible, we require automated unsupervised models to detect outliers. This project aims to build a robust detection system by performing a comparative analysis of three algorithms: Isolation Forest, Local Outlier Factor (LOF), and One-Class SVM. By evaluating these models, we seek to identify the architecture that most accurately distinguishes between "Normal" data and "Anomalies," ensuring proactive security and data integrity.

### Pipeline

#### 1. Problem Definition
- **Classification Task:** Determine if a data point is Normal (1) or an Anomaly (-1).
- **Objective:** Perform a detailed comparative analysis across Isolation Forest, Local Outlier Factor, and One-Class SVM to identify the most reliable architecture for outlier detection.

#### 2. Data Understanding
- **Dataset Content:** A collection of 500 samples consisting of features like Transaction Amount, Frequency, and Time.
- **Input Features (X):** Numerical data representing behavioral patterns (e.g., Feature_1, Feature_2).
- **Target (y):** Labels where 1 represents a "Normal" observation and -1 represents an "Anomaly" (Outlier).

#### 3. Data Preprocessing
- **Cleaning:** Remove non-predictive columns and ensure the data is formatted for unsupervised learning.
- **Standardization:** Apply a `StandardScaler` to normalize numerical features, ensuring features with larger ranges (like transaction amounts) do not disproportionately influence distance-based models like LOF and SVM.

#### 4. Train-Test Split
- **Data Allocation:** Training set (80%) to establish the "normal" baseline and Test set (20%) to validate detection accuracy.
- **Purpose:** Allows measurement of the model's ability to generalize and identify anomalies in unseen data.

#### 5. Model Selection and Training
Three different anomaly detection models are initialized and trained on the same data for a fair comparison:
- **Isolation Forest:** A tree-based model that explicitly isolates anomalies by randomly selecting a feature and a split value.
- **Local Outlier Factor (LOF):** A density-based method that identifies outliers by comparing the local density of a point to its neighbors.
- **One-Class SVM:** Learns a decision boundary that encompasses the "normal" data points in a high-dimensional space.

#### 6. Predictions
- **Execution:** Each model classifies the 100 unseen samples in the test set.
- **Output Generation:** Prediction arrays where each algorithm flags points as either 1 (Normal) or -1 (Anomaly).

#### 7. Model Evaluation and Comparison

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| **Isolation Forest** | **0.96** | **0.70** | **0.70** | **0.70** |
| Local Outlier Factor | 0.88 | 0.00 | 0.00 | 0.00 |
| One-Class SVM | 0.93 | 0.41 | 0.43 | 0.42 |

**Isolation Forest is the best model** among all tested as it achieved the highest accuracy and balanced precision and recall, making it the most effective at isolating outliers without misclassifying normal data.

#### 8. Model Saving
- All trained models, along with the anomaly scaler, are saved using the Pickle library into permanent `.pkl` files.
- These files allow for immediate deployment in a fraud detection system for real-time filtering without the need for retraining.

---

## Experiment 8 — Student Performance Level Analysis

### Aim
Implement Multiclass Classification models for Student Performance Level analysis and perform comparative analysis. (Random Forest, Decision Tree, Multinomial Logistic Regression, XGBoost, K-Nearest Neighbors)

### Pipeline

#### 1. Problem Definition
The objective of this experiment is to develop an automated system to predict a student's **Performance Level** based on various academic and behavioral metrics.
- **Classification Task:** A multiclass classification problem where students are categorized into distinct levels (e.g., 0, 1, or 2).
- **Objective:** Perform a comparative analysis across five distinct algorithms — Multinomial Logistic Regression, Decision Tree, Random Forest, Gradient Boosting (XGBoost/GBM), and K-Nearest Neighbors (KNN) — to determine which model best predicts academic outcomes.

#### 2. Data Understanding
- **Dataset Content:** A collection of 500 student samples with academic indicators.
- **Input Features (X):**
  - **Numerical:** Study Hours, Attendance Percentage, Assignment Score, Internal Marks
  - **Categorical:** Participation (Low/Medium/High), Internet Access (Yes/No), Previous Grade (A/B/C)
- **Target (y):** Performance_Level (Multiclass labels representing different tiers of academic achievement)

#### 3. Data Preprocessing
- **Categorical Encoding:** Columns such as Participation, Internet_Access, and Previous_Grade were converted into numerical format using `pd.get_dummies(drop_first=True)`.
- **Feature-Target Separation:** Input Features (X) and Target (y): the `Performance_Level` label.
- **Standardization:** `StandardScaler` was applied to normalize the numerical features.

#### 4. Train-Test Split
- **Data Allocation:** Training set (80%) and Test set (20%).
- **Random State:** A fixed `random_state=42` was used to ensure reproducibility across different runs.

#### 5. Model Selection and Training
Five different classification architectures were initialized and trained:
1. **Multinomial Logistic Regression:** A linear model adapted for multiclass settings using the softmax function.
2. **Decision Tree:** A non-linear model that splits data based on feature thresholds.
3. **Random Forest:** An ensemble of decision trees that reduces overfitting by averaging multiple "votes."
4. **Gradient Boosting:** An iterative ensemble technique that builds new trees to correct errors made by previous ones.
5. **K-Nearest Neighbors (KNN):** A distance-based model that classifies points based on the majority label of their nearest neighbors.

#### 6. Predictions
- The standardized test set was processed by all five models, transforming academic and behavioral input patterns into specific student performance categories (0, 1, or 2).
- The predicted output is stored to further use in model evaluation.

#### 7. Model Evaluation and Comparison
The models were evaluated using **Accuracy**, **Precision**, **Recall**, and **F1-Score** (using 'macro' averaging to account for class distribution) as well as AUC-ROC curve.

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.9700 | 0.6474 | 0.6599 | 0.6533 |
| Decision Tree | 0.9000 | 0.9320 | 0.9320 | 0.9320 |
| Random Forest | 0.9400 | 0.6269 | 0.6395 | 0.6331 |
| **Gradient Boosting** | **0.9500** | **0.9661** | **0.9660** | **0.9660** |
| K-Nearest Neighbors | 0.7900 | 0.5272 | 0.5374 | 0.5321 |

**ROC-AUC Curve Analysis:**

| Model | Class 0 | Class 1 | Class 2 |
|-------|---------|---------|---------|
| Logistic Regression | 1.00 | 0.96 | 0.86 |
| Decision Tree | 0.90 | 0.90 | 1.00 |
| Random Forest | 1.00 | 0.98 | 1.00 |
| **Gradient Boosting** | **0.99** | **0.99** | **1.00** |
| K-Nearest Neighbors | 0.92 | 0.89 | 0.50 |

**Conclusion:** While Logistic Regression achieved the highest raw accuracy, **Gradient Boosting** is the most robust model for this dataset, as it provides the best balance between high accuracy and superior Precision/Recall/F1-Scores.

#### 8. Model Saving
- All five trained models and the StandardScaler were serialized into `.pkl` files using the Pickle library.
- These files allow the prediction system to be reloaded in a production environment (like a school dashboard) without retraining the models.

---

## Experiment 9 — Physiological Signal Classification

### Aim
Implement a classification model to distinguish between normal and abnormal physiological signals using extracted signal features. (Perform comparative analysis on different ML models)

### Pipeline

#### 1. Problem Definition
The primary goal is to automate the detection of abnormal heart rhythms (arrhythmias) which is critical for early diagnosis of cardiovascular diseases.
- **Classification Task:** Categorize heartbeat samples into "Normal" (0) and "Abnormal" (1).
- **Objective:** Perform a comparative analysis using a Decision Tree, Random Forest, and SVM, and ultimately combine them into a Voting Classifier to improve overall prediction stability and accuracy.

#### 2. Data Understanding
- **Dataset Content:** ECG recordings from the MIT-BIH and PTB Diagnostic Datasets, containing thousands of heartbeat samples.
- **Input Features (X):** 187 numerical features representing the normalized intensity of the ECG signal over a single heartbeat period.
- **Target (y):** Binary labels where 0 represents a Normal heartbeat and 1 represents an Abnormal heartbeat.

#### 3. Data Preprocessing
- **Data Integration:** Combined separate CSV files for normal and abnormal cases into a single unified dataframe.
- **Feature Extraction:** Separated the signal data (first 187 columns) from the ground truth labels (last column).
- **Class Labeling:** Explicitly assigned numerical values (0 and 1) to distinguish between the two health states.
- **Stratification:** Ensured that the distribution of normal vs. abnormal cases remains consistent during the split to handle class imbalances.

#### 4. Train-Test Split
- **Data Allocation:** Training set (80%) and Test set (20%) for final validation.
- **Random State:** A fixed seed (42) was used to ensure reproducibility.

#### 5. Model Selection and Training
Four distinct configurations were trained to compare individual performance vs. ensemble performance:
- **Decision Tree:** A baseline model using 'entropy' as the split criterion with a maximum depth of 10.
- **Random Forest:** An ensemble of 100 trees to reduce variance and improve accuracy through bagging.
- **Support Vector Machine (SVM):** A high-dimensional classifier used to find the optimal hyperplane between normal and abnormal signals.
- **Voting Classifier (Soft Voting):** A meta-classifier that aggregates the probability predictions of the three models above to make a final "consensus" decision.

#### 6. Predictions
- Each trained model was used to predict the labels for the unseen samples in the test set.
- The Voting Classifier calculated the average probability across all base models to determine the most likely class for each heartbeat.

#### 7. Model Evaluation and Comparison

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Decision Tree | 0.8952 | 0.9355 | 0.9182 | 0.9268 |
| **Random Forest** | **0.9705** | **0.9701** | **0.9895** | **0.9797** |
| SVM | 0.9024 | 0.9139 | 0.9548 | 0.9339 |
| Voting Ensemble | 0.9443 | 0.9567 | 0.9667 | 0.9617 |

**AUC-ROC Score:**

| Decision Tree | Random Forest | SVM | Voting Ensemble |
|---------------|---------------|-----|-----------------|
| 0.93 | 0.99 | 0.94 | 0.98 |

**Random Forest is the best individual model** with the highest accuracy (97.05%) and F1-score (0.9797). The Voting Ensemble provides a strong balanced alternative.

#### 8. Model Saving
- The trained Voting Classifier was saved into a `.pkl` file using the Pickle library.
- This file allows medical software to load the model and analyze live ECG streams without needing to retrain the algorithms.

---

## Experiment 10 — Iris Classification

### Aim
Iris Flower Classification: Use the Iris dataset to build a classification model that predicts the species of iris flowers. [Dataset: Load dataset from sklearn]

### Pipeline

#### 1. Problem Definition
The primary goal is to develop a robust automated classification system to identify biological species based on physical measurements.
- **Classification Task:** Determine if a data point belongs to the **Setosa**, **Versicolor**, or **Virginica** class.
- **Objective:** Perform a detailed comparative analysis across three distinct algorithms — Decision Tree, K-Nearest Neighbors (KNN), and Logistic Regression — to identify the most accurate architecture for multiclass classification.

#### 2. Data Understanding
- **Dataset Content:** The classic Iris dataset consisting of 150 samples with four physical attributes: sepal length, sepal width, petal length, and petal width.
- **Input Features (X):** Numerical data representing the four flower measurements.
- **Target (y):** Labels where 0, 1, and 2 represent the three specific species of iris.

#### 3. Data Preprocessing
- **Data Formatting:** Separate the dataset into a feature matrix (X) and a target vector (y).
- **Cleanliness:** As the Scikit-Learn iris dataset is a standard benchmark, the data is verified to be clean and correctly formatted for supervised learning.

#### 4. Train-Test Split
- **Data Allocation:** Training set (80%) and Test set (20%) to validate accuracy.
- **Reproducibility:** A `random_state` of 42 ensures the data split remains consistent across different experimental runs.

#### 5. Model Selection and Training
Three different classification models are initialized and trained on the same data for a fair comparison:
- **Decision Tree Classifier:** A tree-based model that breaks down the dataset into smaller subsets while an associated decision tree is incrementally developed.
- **K-Nearest Neighbors (KNN):** An instance-based method that classifies a point based on the majority class of its nearest neighbors in the feature space.
- **Logistic Regression:** A linear model that learns the probability of a sample belonging to each of the three species classes.

#### 6. Predictions
- **Execution:** Each model classifies the 30 unseen samples in the test set.
- **Output Generation:** Prediction arrays where each algorithm flags points as either class 0, 1, or 2.

#### 7. Model Evaluation and Comparison

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Decision Tree | 1.00 | 1.00 | 1.00 | 1.00 |
| KNN | 1.00 | 1.00 | 1.00 | 1.00 |
| Logistic Regression | 1.00 | 1.00 | 1.00 | 1.00 |

As the data is very simple and all models achieved perfect scores, the AUC-ROC of all models was also **1.00**.

#### 8. Model Saving
- The top-performing trained model (Logistic Regression) is saved using the Pickle library into a permanent `.pkl` file named `logistic_iris.pkl`.
- This allows for immediate deployment in an automated species classification system without retraining.

---

## Experiment 11 — Diabetes Prediction using Ensemble Learning

### Aim
To predict whether a person has diabetes based on features such as blood pressure, skin thickness, age, etc., using the bagging ensemble technique. Also perform comparative analysis among the Bagging Classifier, Random Forest, and the Decision Tree Classifier.

### Pipeline

#### 1. Problem Definition
The goal is to implement an automated classification system to predict diabetes outcomes based on diagnostic metrics.
- **Classification Task:** A binary classification problem where samples are categorized into two distinct classes (Outcome 0 or 1).
- **Objective:** Perform a comparative analysis between a single Decision Tree and ensemble methods (Random Forest and Bagging) to determine which architecture offers the most reliable predictive performance.

#### 2. Data Understanding
- **Input Features (X):** Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age.
- **Target (y):** Outcome — Binary labels where 1 indicates presence of diabetes and 0 indicates absence.

#### 3. Data Preprocessing
- **Missing Value Handling:** Columns containing invalid zero values (Glucose, BloodPressure, SkinThickness, Insulin, BMI) were identified. These zeros were replaced with NaN and subsequently imputed using the median value of each column.
- **Feature-Target Separation:** Dataset divided into input features (X) and the target variable (y).
- **Standardization:** `StandardScaler` was applied to normalize the feature set, ensuring variables with different scales do not bias the algorithms.

#### 4. Train-Test Split
- **Data Allocation:** Training set (80%) and Test set (20%) to validate performance on unseen data.
- **Random State:** A fixed `random_state=42` was used to ensure reproducibility.

#### 5. Model Selection and Training
Three distinct classification architectures were initialized and trained:
- **Decision Tree:** A base non-linear model that creates decision rules based on feature thresholds.
- **Random Forest:** An ensemble technique that builds multiple decision trees on different data subsets and averages their results to improve accuracy and control overfitting.
- **Bagging Classifier:** An ensemble method using a `DecisionTreeClassifier` as the base estimator, training 200 different versions of the model on bootstrapped samples to reduce variance.

#### 6. Predictions
- The standardized test set was processed by all three trained models to generate predicted labels.
- The models transformed medical input patterns into binary classifications (0 or 1), compared against actual ground-truth outcomes.

#### 7. Model Evaluation and Comparison

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Decision Tree | 0.71 | 0.59 | 0.61 | 0.60 |
| Random Forest | 0.74 | 0.63 | 0.65 | 0.64 |
| **Bagging (Decision Tree)** | **0.77** | **0.67** | **0.70** | **0.69** |

**AUC-ROC Score:**

| Decision Tree | Random Forest | Bagging |
|---------------|---------------|---------|
| 0.69 | 0.83 | 0.83 |

**The Bagging Classifier outperforms** Random Forest and Decision Tree across all metrics, demonstrating the value of bootstrap aggregating for variance reduction on medical data.

#### 8. Model Saving
- All three trained models (Decision Tree, Random Forest, and Bagging Classifier) were exported into `.pkl` files using the Pickle library.
- These files allow for the immediate reloading of the predictive system into a production environment for real-time analysis.

---

## Experiment 12 — L1 & L2 Regularization on Melbourne Housing Dataset

### Aim
To implement L1 (Lasso) and L2 (Ridge) regularization techniques on the Melbourne Housing dataset to predict house prices and analyze their performance.

### Pipeline

#### 1. Problem Definition
The primary goal is to develop a predictive model for real estate valuation using advanced regression techniques.
- **Regression Task:** Predict the continuous numerical value of house prices based on various structural and locational features.
- **Objective:** Perform a comparative analysis between Standard Linear Regression, Lasso (L1), and Ridge (L2) regularization to determine which approach best handles multicollinearity and prevents model over-complexity.

#### 2. Data Understanding
- **Dataset Content:** The Melbourne House Price dataset, containing records of real estate transactions.
- **Input Features (X):**
  - **Numerical:** Rooms, Distance, Bedroom2, Bathroom, Car, Landsize, BuildingArea, YearBuilt, Propertycount
  - **Categorical:** Suburb, Address, Type, Method, SellerG, Date, CouncilArea, Regionname
- **Target (y):** Price — The market value of the property in AUD.

#### 3. Data Preprocessing
- **Handling Missing Values:** Null entries in features like 'Car', 'BuildingArea', and 'YearBuilt' are addressed using zero-filling or mean/median imputation.
- **Categorical Encoding:** One-Hot Encoding (`get_dummies`) is applied to transform non-numeric features into binary columns compatible with regression algorithms.
- **Feature-Target Separation:** Dataset divided into the feature matrix (X) and the target variable (y).

#### 4. Train-Test Split
- **Data Allocation:** Training set (80%) to fit the model and Test set (20%) to evaluate performance on unseen data.
- **Random State:** A fixed `random_state` is used to ensure reproducibility.

#### 5. Model Selection and Training
Three different regression architectures are trained to observe the impact of penalties:
- **Linear Regression:** A baseline model that calculates coefficients without any regularization constraints.
- **Lasso Regression (L1):** Adds an absolute value penalty to the loss function, which can shrink some coefficients to exactly zero, performing automatic feature selection.
- **Ridge Regression (L2):** Adds a squared magnitude penalty to the loss function, which helps manage multicollinearity by shrinking coefficients without setting them to zero.

#### 6. Predictions
- The trained models (Linear, Lasso, and Ridge) are used to predict house prices for the properties in the test set.
- Predicted values are stored to calculate error metrics and compare against actual house prices.

#### 7. Model Evaluation and Comparison

**R² Score Comparison:**

| Linear Regression | Ridge (L2) | Lasso (L1) |
|-------------------|------------|------------|
| 0.6426 | 0.6529 | 0.6530 |

**MSE and R² Detailed Comparison:**

| Model | MSE | R² Score |
|-------|-----|----------|
| Linear Regression | — | 0.6426 |
| Lasso (L1) | 161,700,900,000 | 0.5791 |
| **Ridge (L2)** | **128,873,000,000** | **0.6646** |

**Ridge (L2) achieves the best balance** between MSE and R², demonstrating that squared penalty regularization is more effective than absolute penalty on this dataset. Lasso also marginally improves upon baseline by applying automatic feature selection.

#### 8. Model Saving
- The optimized regularized model and preprocessing objects are serialized into `.pkl` files using the Pickle library.
- These files allow the prediction system to be reloaded for real-time house price estimations without needing to retrain the model.

---

## Experiment 13 — Cardiovascular Disease Risk Prediction

**Aim:** Develop a machine learning pipeline that predicts the risk of cardiovascular disease using clinical data from two datasets: the Cleveland Heart Disease dataset and the Pima Indians Diabetes dataset. The project applies and compares multiple regression and classification models.

### Problem Statement

Cardiovascular disease (CVD) is one of the leading causes of mortality worldwide. Early and accurate identification of individuals at risk can significantly reduce fatality rates by enabling timely medical intervention.

Traditional diagnostic approaches depend heavily on specialist judgment and may be inconsistent across different clinical settings. This project addresses the problem by building a data-driven risk assessment system using structured clinical features such as age, blood pressure, cholesterol levels, and heart rate.

The system uses:
- Regression models to generate a continuous risk score
- Classification models to provide a binary diagnosis (Yes/No)

---

### Datasets Used

| Dataset | Source | Records | Features | Target Variable |
|---------|--------|---------|----------|--------------------|
| Cleveland Heart Disease | UCI ML Repository | 303 | 13 | Heart disease (0/1) |
| Pima Indians Diabetes | Kaggle | 768 | 8 | Diabetes outcome (0/1) |

---

### Target Variables

- **Cleveland Dataset**
  - `target`: 0 = No disease, 1 = Disease present
  - Derived from `num` column (num > 0 → 1)

- **Diabetes Dataset**
  - `Outcome`: 0 = Non-diabetic, 1 = Diabetic

---

### Data Statistics

#### Cleveland Heart Disease Dataset

| Feature | Mean | Std Dev | Min | Max |
|---------|------|---------|-----|-----|
| Age | 54.4 | 9.0 | 29 | 77 |
| Resting Blood Pressure | 131 | 17 | 94 | 200 |
| Cholesterol | 246 | 51 | 126 | 564 |
| Max Heart Rate | 150 | 22 | 71 | 202 |

#### Pima Indians Diabetes Dataset

| Feature | Mean | Std Dev | Min | Max |
|---------|------|---------|-----|-----|
| Glucose | 120 | 32 | 0 | 199 |
| Blood Pressure | 69 | 19 | 0 | 122 |
| BMI | 32 | 7.9 | 0 | 67 |
| Insulin | 80 | 115 | 0 | 846 |

#### Missing Values Handling

| Dataset | Columns Affected | Handling Method |
|---------|-----------------|-----------------|
| Cleveland | '?' values | Converted to NaN |
| Diabetes | Glucose, BP, SkinThickness, Insulin, BMI | 0 replaced with NaN |
| Both | All columns | Median imputation |

---

### Machine Learning Pipeline

#### 1. Data Loading and Understanding

- Data loaded using `pd.read_csv()`
- Missing values handled
- Target column binarised for Cleveland dataset
- `df.info()` and `df.describe()` used for analysis

#### 2. Data Preprocessing

- Invalid zero values replaced with NaN (Diabetes dataset)
- Missing values filled using median imputation
- Feature-target split into X and y
- Train-test split (80:20) with stratification
- Feature scaling using `StandardScaler`

#### 3. Models Used

**Regression Models:**
- Linear Regression
- Ridge Regression (alpha = 1)
- Lasso Regression (alpha = 0.1)

**Classification Models:**
- Logistic Regression (L1 regularization)
- Logistic Regression (L2 regularization)
- Decision Tree (max depth = 5)
- Random Forest

#### 4. Predictions

- Predictions generated using `model.predict()`
- Probabilities obtained using `predict_proba()` (Random Forest)
- Supports custom input for real-world inference after scaling

#### 5. Visualizations

- RMSE comparison (regression models)
- Accuracy and F1-score comparison (classification models)
- Feature importance (Random Forest)
- Confusion matrix heatmap
- User vs dataset feature comparison
- Risk score visualization with thresholds (0.3 and 0.6)
- Disease probability bar chart

---

### Model Evaluation

#### Regression Results

| Model | RMSE (Cleveland) | RMSE (Diabetes) | R² Score |
|-------|-----------------|-----------------|----------|
| Linear Regression | 0.42 | 0.45 | 0.30 |
| Ridge Regression | 0.41 | 0.44 | 0.32 |
| Lasso Regression | 0.43 | 0.46 | 0.28 |

#### Classification Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression (L1) | 0.82 | 0.80 | 0.78 | 0.79 |
| Logistic Regression (L2) | 0.83 | 0.81 | 0.80 | 0.80 |
| Decision Tree | 0.78 | 0.75 | 0.76 | 0.75 |
| **Random Forest** | **0.86** | **0.84** | **0.85** | **0.84** |

**Random Forest is the best-performing model** across both accuracy and F1-score, demonstrating strong generalization capability on clinical data.
