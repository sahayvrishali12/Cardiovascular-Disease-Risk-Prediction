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
| 13 | Cardiovascular Disease Risk Prediction ( Lab Mini Project ) | 09-03-2026 |

---

## Experiment 1 — Data Preprocessing

**Aim:** Text pre-processing in IPL dataset using Python libraries.

### Problem Statement

Raw datasets, especially those derived from real-world events like IPL cricket matches, often contain inconsistencies, missing values, duplicate entries, and unstructured text that cannot be directly used for analysis or model training. This experiment focuses on cleaning and transforming the IPL dataset into a structured, analysis-ready format using standard Python libraries.

### Dataset

The IPL (Indian Premier League) dataset contains match-level and player-level information including team names, venues, player names, dismissal types, and match results. These fields are primarily textual and require preprocessing before any downstream analysis or machine learning can be applied.

### Libraries Used

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

### Key Takeaways

- Raw IPL data contained significant missing values in player and dismissal-related fields, which were handled using context-aware imputation strategies.
- Standardizing team names and removing inconsistencies was critical to ensure groupby and aggregation operations produced accurate results.
- One-hot encoding of categorical features ensured compatibility with machine learning algorithms in later experiments.
- The preprocessing pipeline reduced noise in the dataset and improved overall data quality, forming a reliable foundation for analysis and modelling.

---

## Experiment 2 — Predicting Housing Prices

**Aim:** Develop a regression model to predict house prices based on features like location, size, and amenities.

### Problem Statement

Estimating the market value of real estate is a complex task that depends on multiple factors such as property structure, location, and economic conditions. Traditionally, property valuation has relied on manual assessments and the judgment of real estate agents. However, this approach can be inconsistent, time-consuming, and often influenced by human bias. With the growing complexity of the real estate market, there is a need for more accurate and data-driven methods that can deliver reliable valuations quickly.

This project aims to overcome these challenges by automating the valuation process using machine learning techniques. By analyzing patterns within historical housing data, we can build a model that predicts property prices more accurately. Using a dataset of 120 property records, the goal is to develop a regression model that converts property features into precise price estimates. This approach reduces dependency on subjective judgment and provides a consistent, transparent, and data-driven solution for buyers and sellers.

### Input Features

- **Location:** Type of area (Rural, Urban, Suburban)
- **Area_sqft:** Total size of the property in square feet
- **Bedrooms/Bathrooms:** Number of rooms in the house
- **Balcony/Parking:** Availability of additional facilities
- **Age_Years:** Age of the property (reflects depreciation or value)
- **Furnished:** Whether the house is furnished (Yes/No)

### Target Variable

- **Price:** Estimated value of the house

### Model Pipeline

#### 1. Data Understanding

The target variable (y) is the house price, which the model aims to predict.

Input features (X) include:
- Numerical: Area_sqft, Bedrooms, Bathrooms, Age_Years, Parking, Balconies
- Categorical: Location and Furnished

The dataset is loaded using `pd.read_csv()` and checked using `df.info()` to understand structure and data types.

#### 2. Data Preprocessing

- Encoding categorical variables: Text values like "Urban" or "Rural" are converted into numerical form using One-Hot Encoding (`pd.get_dummies()`), creating columns such as `Location_Urban` and `Furnished_Yes`.
- Splitting features and target:
  - X → input features
  - y → output (Price)

#### 3. Train–Test Split

The dataset is divided using `train_test_split()`:
- 80% training data for learning
- 20% testing data for evaluation

#### 4. Model Selection and Training

Linear Regression is chosen to model the relationship between features and price. The model is trained using `model.fit()`, where it learns coefficients that minimize prediction error.

#### 5. Predictions

The trained model is used on test data via `model.predict()`. Predicted values are compared with actual values to evaluate performance. A graph is plotted to visually analyze prediction accuracy and trends.

#### 6. Model Evaluation

Performance is measured using:
- **Mean Squared Error (MSE):** Lower value indicates better accuracy
- **R² Score:** Indicates how well features explain price variation

#### 7. Model Saving

The trained model is saved using the Pickle library (`pickle.dump()`). This allows future use without retraining by loading it using `pickle.load()`.

### Key Takeaways

- The linear regression model successfully learned the relationship between housing features such as location, area, amenities, and age with the house price, producing stable and consistent predictions.
- The model achieved very low mean squared error and a high R² score, indicating strong predictive performance. This high accuracy is expected since the dataset is synthetic and well-structured.
- One-hot encoding of categorical variables and standardization of numerical features helped ensure that all input features contributed fairly to the regression model.
- Saving both the trained model and the scaler ensures the model can be reliably reused for future predictions on new housing data.

---

## Experiment 3 — Stock Price Prediction

**Aim:** Develop a time series prediction model to forecast stock prices.

### Problem Overview

The stock market is a highly dynamic system where price movements are influenced by various factors such as historical trends, investor sentiment, and broader economic conditions. Traditional forecasting methods often depend on human analysis, which can be inconsistent due to bias or difficulty in handling large volumes of sequential data. Therefore, there is a need for automated approaches that can efficiently analyze patterns over time.

This project focuses on developing a data-driven forecasting model that captures hidden temporal patterns within historical stock data. By analyzing past price movements, the model aims to predict future stock prices accurately. Instead of relying on subjective judgments, the approach uses historical "lag" values to build a reliable and consistent prediction system. This ensures that the forecasting process is based on actual data patterns, making it more transparent and useful for investors and analysts.

### Model Pipeline

#### 1. Dataset Understanding

The dataset consists of time-based records, where each entry represents stock data at a specific point in time.

- **Target Variable (y):** Price – the value we aim to predict for the current day.
- **Input Features (X):**
  - Numerical (Lag Features): Prices from the previous 5 days (lag1 to lag5)
  - Categorical: Stock identifier (e.g., AAPL)
- **Excluded Features:** Current-day attributes such as Open, High, Low, Close, and Volume are excluded to ensure the model relies only on past data and avoids data leakage.

#### 2. Data Preprocessing

- **Time Alignment and Encoding:**
  - The Date column is converted using `pd.to_datetime()` and sorted chronologically.
  - Stock names are encoded into numerical format using One-Hot Encoding (`pd.get_dummies()`), allowing the model to distinguish between different stocks.
- **Lag Feature Creation:**
  - Five lag features are generated using the `shift()` function.
  - These features represent stock prices from the previous five days, helping capture time-based patterns.
- **Feature-Target Split:**
  - X contains lag features and encoded stock variables.
  - y contains the target variable (Price).

#### 3. Train–Test Split

Since the data is time-series, shuffling is avoided. The dataset is split chronologically:
- First 80% → Training data
- Last 20% → Testing data

#### 4. Model Selection and Training

A Linear Regression model is selected to model the relationship between past and future prices. Using the `fit()` method, the model learns coefficients that determine the influence of each lag feature on the predicted price.

#### 5. Predictions

Predictions are generated on the test dataset using `predict()`. Actual and predicted prices are plotted to visually assess how well the model captures market trends.

#### 6. Model Evaluation

Model performance is evaluated using:
- **Mean Absolute Error (MAE):** Measures average prediction error
- **R² Score:** Indicates how well past data explains price variation

Lower MAE and higher R² indicate better model performance.

#### 7. Model Saving

The trained model is saved using the Pickle library (`pickle.dump()`). This allows the model to be reused later without retraining by loading it using `pickle.load()`.

---

## Experiment 4 — Customer Churn Prediction

**Aim:** Customer Churn Prediction: Develop a model to predict customer churn in a subscription-based business.

### Problem Statement

Customer retention is a major concern for service-based businesses, as the cost of acquiring new customers is generally higher than retaining existing ones. Traditional manual methods often fail to capture subtle behavioral patterns that indicate whether a customer is likely to leave. This project focuses on automating the identification of customers at risk of churn by analyzing a dataset of 200 records containing historical usage and contract information.

Unlike regression tasks, this is a classification problem where the goal is to predict a categorical outcome: Churn (1) or No Churn (0). By applying a data-driven approach, businesses can move from reactive strategies to proactive retention. Identifying potential churners in advance helps optimize marketing efforts and improves long-term revenue stability.

### Project Pipeline

#### 1. Data Understanding

The dataset contains 200 customer records with 11 attributes.

- **Target Variable (y):** Churn
- **Numerical Features:** Age, Tenure_Months, MonthlyCharges, TotalCharges, SupportTickets
- **Categorical Features:** Gender, SubscriptionType, PaymentMethod, ContractType

#### 2. Data Preprocessing

- **Feature Selection:** The CustomerID column is removed using `df.drop()` since it does not contribute to prediction.
- **Data Cleaning:** Missing values are handled by removing incomplete rows using `df.dropna()`.
- **Categorical Encoding:** Text-based features are converted into numerical form using One-Hot Encoding (`pd.get_dummies()`).

#### 3. Feature-Target & Train-Test Split

- The dataset is divided into input features (X) and target variable (y).
- **Data Split Ratio:** 80% training and 20% testing
- **Test Set:** 40 unseen records used for evaluating model performance

#### 4. Model Selection and Training

- **Model Chosen:** Logistic Regression (suitable for binary classification)
- **Initialization:** Model is initialized with `max_iter = 1000`
- **Training:** Model is trained using the `fit()` function

#### 5. Predictions

The trained model is applied to the test data using `predict()`. Output: Binary predictions indicating churn (1) or no churn (0) for each customer.

#### 6. Model Evaluation

- **Accuracy:** 75% (30 out of 40 predictions are correct)
- **Classification Report:**
  - Class 0 (No Churn): High precision and recall (~85%)
  - Class 1 (Churn): Lower precision and recall (~29%), indicating difficulty in detecting churn cases

#### 7. Model Saving

The trained model is saved using the Pickle library (`pickle.dump()`). The saved file can be loaded later using `pickle.load()` for future predictions without retraining.

---

## Experiment 5 — Spam Detection

**Aim:** Spam Email Detection: Build a spam email filter using text classification algorithms and perform comparative analysis. (Naive Bayes, Logistic Regression, Support Vector Machine, Decision Tree, Random Forest)

### Problem Statement

Spam detection plays a crucial role in ensuring email security and maintaining user productivity, as unwanted or malicious emails can lead to security risks and cluttered inboxes. Manual filtering is not feasible due to the large volume of emails, making automated classification systems necessary. This project focuses on developing an efficient spam detection system by comparing five different text classification algorithms: Naïve Bayes, Logistic Regression, Support Vector Machine (SVM), Decision Tree, and Random Forest. The objective is to determine which model most accurately classifies emails as "Spam" or "Ham" (legitimate), thereby improving security and resource management.

### Pipeline

#### 1. Problem Definition

The main objective of this experiment is to design an automated spam detection system using text classification techniques.

- **Classification Task:** Identify whether an email is Spam (1) or Ham (0)
- **Objective:** Perform comparative analysis of five algorithms — Naïve Bayes, Logistic Regression, SVM, Decision Tree, and Random Forest — to determine the best-performing model

#### 2. Data Understanding

- **Dataset Content:** A large collection of email messages labeled as spam or legitimate
- **Target Variable (y):** Binary labels where 1 represents spam and 0 represents ham
- **Input Features (X):** Raw email text including subject and body

#### 3. Data Preprocessing

Since machine learning models cannot process raw text directly, the data undergoes cleaning and transformation.

- **Text Cleaning:** Removal of punctuation, numbers, and special characters
- **Stop-word Removal:** Eliminating common words like "and", "the", and "is"
- **Vectorization:** Converting text into numerical form using TF-IDF (Term Frequency-Inverse Document Frequency)

#### 4. Train-Test Split

- **Data Allocation:** 80% data for training and 20% for testing
- **Purpose:** Ensures the model generalizes well and avoids overfitting

#### 5. Model Selection and Training

Five classification models are trained on the same dataset for fair comparison:

- **Naïve Bayes:** Probabilistic model suitable for text data
- **Logistic Regression:** Linear model for binary classification
- **Support Vector Machine (SVM):** Finds optimal boundary between classes
- **Decision Tree:** Uses rule-based splitting for classification
- **Random Forest:** Ensemble of decision trees for improved accuracy

#### 6. Predictions

After training, each model is used to classify unseen test emails. The `predict()` function generates binary outputs for each model. Predictions are compared with actual labels to identify errors such as false positives and false negatives.

#### 7. Model Comparison and Evaluation

The models are evaluated using Accuracy, Precision, and Recall.

| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| Naive Bayes | 0.976 | 1.0 | 0.825 |
| Logistic Regression | 0.967 | 1.0 | 0.758 |
| SVM | 0.984 | 1.0 | 0.880 |
| Decision Tree | 0.968 | 0.9 | 0.859 |
| Random Forest | 0.979 | 1.0 | 0.830 |

Accuracy measures overall correctness, where SVM performs best. Precision indicates correctness of positive predictions, where most models achieve perfect scores. Recall measures the ability to detect actual spam, where SVM achieves the highest value. Overall, **SVM is the best-performing model** due to its high accuracy, recall, and precision.

---

## Experiment 6 — Credit Risk Assessment

**Aim:** Build a credit scoring model to assess the creditworthiness of applicants using historical financial data and perform comparative analysis (Logistic Regression, Random Forest, XGBoost).

### Problem Definition

Automated credit scoring is vital for financial institutions to mitigate potential losses and optimize lending decisions. Because the volume of applications makes manual review inefficient, we require robust classification models to predict applicant creditworthiness. This project aims to build a reliable credit filter by performing a comparative analysis of three algorithms: Logistic Regression, Random Forest, and Gradient Boosting. By evaluating these models, we seek to identify the architecture that most accurately categorizes "High," "Medium," and "Low" risk applicants, ensuring proactive risk management and improved financial stability.

### Pipeline

#### 1. Problem Definition

The primary goal of our experiment is to develop a robust automated credit scoring model using advanced classification techniques:

- **Classification Task:** Determine the creditworthiness of an applicant, categorizing them into risk levels such as High, Medium, or Low.
- **Objective:** Perform a detailed comparative analysis across three distinct algorithms — Logistic Regression, Random Forest, and Gradient Boosting — to identify the most reliable architecture for financial deployment.

#### 2. Data Understanding

- **Dataset Content:** A collection of 500 samples of applicant profiles including historical financial health indicators.
- **Input Features (X):** Raw data including age, annual income, employment years, credit score, loan amount, and late payments.
- **Target (y):** Categorical labels representing "Credit Risk" (High, Medium, or Low).

#### 3. Data Preprocessing

Since financial data requires specific cleaning, we put it through a rigorous transformation process:

- **Feature Selection:** Drop non-predictive identifiers, such as `Applicant_ID`, to focus on significant data.
- **Label Encoding:** Utilize a `LabelEncoder` to transform categorical risk labels into numerical vectors.
- **Standardization:** Apply a `StandardScaler` to normalize numerical features like income and loan amounts, preventing large values from dominating the model logic.

#### 4. Train-Test Split

To ensure our filter works on applicants it has never seen before, we strategically divide the dataset:

- **Data Allocation:** Split the cleaned dataset into a Training set (80%) to build our models and a Test set (20%) to validate them.
- **Purpose:** Prevent "overfitting" and allow measurement of how accurately the filter will perform in a live financial environment.

#### 5. Model Selection and Training

We initialize and train three different classification models on the same data to ensure a fair comparison:

- **Logistic Regression:** A baseline linear model to estimate risk probabilities based on weighted input features.
- **Random Forest:** An ensemble method that constructs multiple decision trees, improving stability and capturing non-linear relationships.
- **Gradient Boosting:** Corrects errors of previous trees sequentially to achieve high-fidelity predictions.

#### 6. Predictions

- **Execution:** Each model classifies the 100 unseen samples in the test set.
- **Output Generation:** Binary and multi-class prediction arrays are generated for each algorithm to prepare for the evaluation phase.

#### 7. Model Evaluation and Comparison

The performance of each model is measured using Accuracy, Precision, Recall, and F1.

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Logistic Regression | 0.85 | 0.81 | 0.76 | 0.79 |
| Random Forest | 0.98 | 0.98 | 0.87 | 0.91 |
| **Gradient Boosting** | **1.00** | **1.00** | **1.00** | **1.00** |

**Gradient Boosting is the best model** among all tested as it scored the highest across all critical financial metrics and showed perfect classification capability on the test set.

---

## Experiment 7 — Anomaly Detection

**Aim:** Implement an anomaly detection system for detecting outliers in data (e.g., fraud detection) and perform comparative analysis. (Isolation Forest, Local Outlier Factor, One-Class SVM)

### Problem

Anomaly detection is critical for identifying rare items, events, or observations which raise suspicions by differing significantly from the majority of the data. Because the volume of transactions makes manual inspection for fraud or errors impossible, we require automated unsupervised models to detect outliers. This project aims to build a robust detection system by performing a comparative analysis of three algorithms: Isolation Forest, Local Outlier Factor (LOF), and One-Class SVM. By evaluating these models, we seek to identify the architecture that most accurately distinguishes between "Normal" data and "Anomalies," ensuring proactive security and data integrity.

### Pipeline

#### 1. Problem Definition

The primary goal of our experiment is to develop a robust automated anomaly detection system using advanced unsupervised learning techniques:

- **Classification Task:** Determine if a data point is Normal (1) or an Anomaly (-1).
- **Objective:** Perform a detailed comparative analysis across three distinct algorithms — Isolation Forest, Local Outlier Factor, and One-Class SVM — to identify the most reliable architecture for outlier detection.

#### 2. Data Understanding

- **Dataset Content:** A collection of 500 samples consisting of features like Transaction Amount, Frequency, and Time.
- **Input Features (X):** Numerical data representing behavioral patterns (e.g., Feature_1, Feature_2).
- **Target (y):** Labels where 1 represents a "Normal" observation and -1 represents an "Anomaly" (Outlier).

#### 3. Data Preprocessing

Since anomaly detection models are sensitive to the scale of data, we put it through a rigorous transformation process:

- **Cleaning:** Remove non-predictive columns and ensure the data is formatted for unsupervised learning.
- **Standardization:** Apply a `StandardScaler` to normalize numerical features, ensuring that features with larger ranges (like transaction amounts) do not disproportionately influence the distance-based models like LOF and SVM.

#### 4. Train-Test Split

To evaluate how well the system identifies new outliers, we strategically divide the dataset:

- **Data Allocation:** Split the dataset into a Training set (80%) to establish the "normal" baseline and a Test set (20%) to validate detection accuracy.
- **Purpose:** This division allows measurement of the model's ability to generalize and identify anomalies in unseen data.

#### 5. Model Selection and Training

We initialize and train three different anomaly detection models on the same data to ensure a fair comparison:

- **Isolation Forest:** A tree-based model that explicitly isolates anomalies by randomly selecting a feature and a split value.
- **Local Outlier Factor (LOF):** A density-based method that identifies outliers by comparing the local density of a point to its neighbors.
- **One-Class SVM:** Learns a decision boundary that encompasses the "normal" data points in a high-dimensional space.

#### 6. Predictions

- **Execution:** Each model classifies the 100 unseen samples in the test set.
- **Output Generation:** Prediction arrays are generated where each algorithm flags points as either 1 (Normal) or -1 (Anomaly).

#### 7. Model Evaluation and Comparison

The performance of each model is measured using Accuracy, Precision, Recall, and F1-Score.

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| **Isolation Forest** | **0.96** | **0.7** | **0.7** | **0.7** |
| Local Outlier Factor | 0.88 | 0 | 0 | 0 |
| One-Class SVM | 0.93 | 0.41 | 0.43 | 0.42 |

**Isolation Forest is the best model** among all tested as it achieved the highest accuracy and balanced precision and recall, making it the most effective at isolating outliers without misclassifying normal data.

---

## Experiment 8 — Student Performance Level Analysis

**Aim:** Implement Multiclass Classification models for Student Performance Level analysis and perform comparative analysis. (Random Forest, Decision Tree, Multinomial Logistic Regression, XGBoost, K-Nearest Neighbors)

### 1. Problem Definition

The objective of this experiment is to develop an automated system to predict a student's **Performance Level** based on various academic and behavioral metrics.

- **Classification Task:** This is a multiclass classification problem where students are categorized into distinct levels (e.g., 0, 1, or 2).
- **Objective:** Perform a comparative analysis across five distinct algorithms — Multinomial Logistic Regression, Decision Tree, Random Forest, Gradient Boosting (XGBoost/GBM), and K-Nearest Neighbors (KNN) — to determine which model best predicts academic outcomes.

### 2. Data Understanding

- **Dataset Content:** A collection of 500 student samples with academic indicators.
- **Input Features (X):**
  - Numerical: Study Hours, Attendance Percentage, Assignment Score, Internal Marks
  - Categorical: Participation (Low/Medium/High), Internet Access (Yes/No), Previous Grade (A/B/C)
- **Target (y):** Performance_Level (Multiclass labels representing different tiers of academic achievement)

### 3. Data Preprocessing

To prepare the dataset for multiclass classification, a rigorous transformation process was applied:

- **Categorical Encoding:** Columns such as Participation, Internet_Access, and Previous_Grade were converted into numerical format using `pd.get_dummies(drop_first=True)`.
- **Indicator Creation:** This process creates binary indicator columns, transforming qualitative behavioral patterns into quantitative data points.
- **Feature-Target Separation:** After encoding, the dataset was divided into:
  - Input Features (X): All academic and behavioral features excluding the target
  - Target (y): The `Performance_Level` label
- **Standardization:** `StandardScaler` was applied to normalize the numerical features.

### 4. Train-Test Split

- **Data Allocation:** The dataset was partitioned into a Training set (80%) and a Test set (20%).
- **Random State:** A fixed `random_state=42` was used to ensure reproducibility.

### 5. Model Selection and Training

Five different classification architectures were initialized and trained:

- **Multinomial Logistic Regression:** A linear model adapted for multiclass settings using the softmax function.
- **Decision Tree:** A non-linear model that splits data based on feature thresholds to create a tree-like decision structure.
- **Random Forest:** An ensemble of decision trees that reduces overfitting by averaging multiple "votes."
- **Gradient Boosting:** An iterative ensemble technique that builds new trees to correct the errors made by previous ones.
- **K-Nearest Neighbors (KNN):** A distance-based model that classifies points based on the majority label of their nearest neighbors.

### 6. Predictions

- **Multiclass Execution:** The standardized test set was processed by all five models, transforming academic and behavioral input patterns into specific student performance categories (0, 1, or 2).
- **Logic Mapping:** Each algorithm generated independent prediction vectors, enabling a deep comparative analysis of their unique decision-making frameworks.
- **Performance Tiering:** The resultant labels allowed for a direct, automated comparison against actual outcomes, facilitating the identification of specific academic tiers for targeted educational support.

### 7. Model Evaluation and Comparison

The models were evaluated using Accuracy, Precision, Recall, and F1-Score (using 'macro' averaging to account for class distribution).

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.9700 | 0.6474 | 0.6599 | 0.6533 |
| Decision Tree | 0.9000 | 0.9320 | 0.9320 | 0.9320 |
| Random Forest | 0.9400 | 0.6269 | 0.6395 | 0.6331 |
| **Gradient Boosting** | **0.9500** | **0.9661** | **0.9660** | **0.9660** |
| K-Nearest Neighbors | 0.7900 | 0.5272 | 0.5374 | 0.5321 |

**Conclusion:** While Logistic Regression achieved the highest raw accuracy, **Gradient Boosting** is the most robust model for this dataset, as it provides the best balance between high accuracy and superior Precision/Recall/F1-Scores.

---

## Experiment 9 — Physiological Signal Classification

**Aim:** Implement a classification model to distinguish between normal and abnormal physiological signals using extracted signal features. (Perform comparative analysis on different ML models)

### Pipeline

#### 1. Problem Definition

The primary goal of this experiment is to automate the detection of abnormal heart rhythms (arrhythmias) which is critical for early diagnosis of cardiovascular diseases:

- **Classification Task:** Categorize heartbeat samples into "Normal" (0) and "Abnormal" (1).
- **Objective:** Perform a comparative analysis using a Decision Tree, Random Forest, and SVM, and ultimately combine them into a Voting Classifier to improve overall prediction stability and accuracy.

#### 2. Data Understanding

- **Dataset Content:** The dataset consists of ECG recordings from the MIT-BIH and PTB Diagnostic Datasets, containing thousands of heartbeat samples.
- **Input Features (X):** 187 numerical features representing the normalized intensity of the ECG signal over a single heartbeat period.
- **Target (y):** Binary labels where 0 represents a Normal heartbeat and 1 represents an Abnormal heartbeat.

#### 3. Data Preprocessing

To prepare the medical signal data for the machine learning models:

- **Data Integration:** Combined separate CSV files for normal and abnormal cases into a single unified dataframe.
- **Feature Extraction:** Separated the signal data (first 187 columns) from the ground truth labels (last column).
- **Class Labeling:** Explicitly assigned numerical values (0 and 1) to distinguish between the two health states.
- **Stratification:** Ensured that the distribution of normal vs. abnormal cases remains consistent during the split to handle class imbalances.

#### 4. Train-Test Split

To validate the model's diagnostic reliability on unseen patients:

- **Data Allocation:** The combined dataset was split into a Training set (80%) and a Test set (20%) for final validation.
- **Random State:** A fixed seed (42) was used to ensure reproducibility.

#### 5. Model Selection and Training

Four distinct configurations were trained to compare individual vs. ensemble performance:

- **Decision Tree:** A baseline model using 'entropy' as the split criterion with a maximum depth of 10.
- **Random Forest:** An ensemble of 100 trees to reduce variance and improve accuracy through bagging.
- **Support Vector Machine (SVM):** A high-dimensional classifier used to find the optimal hyperplane between normal and abnormal signals.
- **Voting Classifier (Soft Voting):** A meta-classifier that aggregates the probability predictions of the three models above to make a final "consensus" decision.

#### 6. Predictions

- **Execution:** Each trained model was used to predict the labels for the 2,911 unseen samples in the test set.
- **Ensemble Logic:** The Voting Classifier calculated the average probability across all base models to determine the most likely class for each heartbeat.

#### 7. Model Evaluation and Comparison

The performance was evaluated using Precision, Recall, and F1-Score, specifically focusing on the model's ability to identify abnormal cases correctly.

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Decision Tree | 0.8952 | 0.9355 | 0.9182 | 0.9268 |
| **Random Forest** | **0.9705** | **0.9701** | **0.9895** | **0.9797** |
| SVM | 0.9024 | 0.9139 | 0.9548 | 0.9339 |
| Voting Ensemble | 0.9443 | 0.9567 | 0.9667 | 0.9617 |

---

## Experiment 10 — Iris Classification

**Aim:** Iris Flower Classification: Use the Iris dataset to build a classification model that predicts the species of iris flowers. [Dataset: Load dataset from sklearn]

### Pipeline

#### 1. Problem Definition

The primary goal of this experiment is to develop a robust automated classification system to identify biological species based on physical measurements:

- **Classification Task:** Determine if a data point belongs to the *Setosa*, *Versicolor*, or *Virginica* class.
- **Objective:** Perform a detailed comparative analysis across three distinct algorithms — Decision Tree, K-Nearest Neighbors (KNN), and Logistic Regression — to identify the most accurate architecture for multiclass classification.

#### 2. Data Understanding

- **Dataset Content:** The classic Iris dataset consisting of 150 samples with four physical attributes: sepal length, sepal width, petal length, and petal width.
- **Input Features (X):** Numerical data representing the four flower measurements.
- **Target (y):** Labels where 0, 1, and 2 represent the three specific species of iris.

#### 3. Data Preprocessing

To ensure the data is suitable for algorithmic processing:

- **Data Formatting:** Separate the dataset into a feature matrix (X) and a target vector (y).
- **Cleanliness:** As the Scikit-Learn iris dataset is a standard benchmark, the data is verified to be clean and correctly formatted for supervised learning.

#### 4. Train-Test Split

To evaluate how well the system identifies species in new samples:

- **Data Allocation:** Split the dataset into a Training set (80%) and a Test set (20%) to validate accuracy.
- **Reproducibility:** Use a `random_state` of 42 to ensure that the data split remains consistent across different experimental runs.

#### 5. Model Selection and Training

Three different classification models are initialized and trained on the same data to ensure a fair comparison:

- **Decision Tree Classifier:** A tree-based model that breaks down the dataset into smaller subsets while an associated decision tree is incrementally developed.
- **K-Nearest Neighbors (KNN):** An instance-based method that classifies a point based on the majority class of its nearest neighbors in the feature space.
- **Logistic Regression:** A linear model that learns the probability of a sample belonging to each of the three species classes.

#### 6. Predictions

- **Execution:** Each model classifies the 30 unseen samples in the test set.
- **Output Generation:** Prediction arrays where each algorithm flags points as either class 0, 1, or 2.

#### 7. Model Evaluation and Comparison

The performance of each model is measured using a Confusion Matrix for visual verification and a Classification Report containing Precision, Recall, and F1-Score.

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Decision Tree | 1.00 | 1.00 | 1.00 | 1.00 |
| KNN | 1.00 | 1.00 | 1.00 | 1.00 |
| Logistic Regression | 1.00 | 1.00 | 1.00 | 1.00 |

---

## Experiment 11 — Diabetes Prediction using Ensemble Learning

**Aim:** To predict whether a person has diabetes based on medical attributes using ensemble techniques and compare the performance of Bagging Classifier, Random Forest, and Decision Tree.

### Pipeline

#### 1. Problem Definition

The primary goal is to develop a predictive model for medical diagnosis:

- **Classification Task:** Determine whether a patient is diabetic (1) or non-diabetic (0).
- **Objective:** Perform comparative analysis among Decision Tree, Bagging Classifier, and Random Forest to identify the most accurate model.

#### 2. Data Understanding

- **Dataset Content:** The dataset contains patient health metrics such as glucose level, blood pressure, skin thickness, insulin, BMI, age, etc.
- **Input Features (X):** Numerical medical attributes.
- **Target (y):** Binary outcome (0 = Non-diabetic, 1 = Diabetic).

#### 3. Data Preprocessing

- **Data Formatting:** Separate dataset into feature matrix (X) and target vector (y).
- **Handling Missing Values:** Replace or remove invalid/missing values.
- **Feature Scaling:** Standardize data to improve model performance.

#### 4. Train-Test Split

- **Data Allocation:** Split dataset into Training (80%) and Testing (20%).
- **Reproducibility:** Use `random_state = 42`.

#### 5. Model Selection and Training

- **Decision Tree Classifier:** A tree-based model that splits data based on feature conditions.
- **Bagging Classifier:** Uses multiple Decision Trees trained on random subsets to reduce variance and improve stability.
- **Random Forest Classifier:** An advanced ensemble method combining bagging and feature randomness for better accuracy.

#### 6. Predictions

- **Execution:** Each model predicts diabetes status on test data.
- **Output:** Binary predictions (0 or 1).

#### 7. Model Evaluation and Comparison

Performance is evaluated using Accuracy, Precision, Recall, and F1-score.

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Decision Tree | 0.74 | 0.62 | 0.72 | 0.67 |
| Bagging | 0.74 | 0.63 | 0.69 | 0.66 |
| Random Forest | 0.72 | 0.60 | 0.61 | 0.61 |

---

## Experiment 12 — L1 & L2 Regularization on Melbourne Housing Dataset

**Aim:** To implement L1 (Lasso) and L2 (Ridge) regularization techniques on the Melbourne Housing dataset to predict house prices and analyze their performance.

### Pipeline

#### 1. Problem Definition

The goal is to build a regression model for price prediction:

- **Regression Task:** Predict house prices based on property features.
- **Objective:** Compare Lasso and Ridge regression to analyze the effect of regularization.

#### 2. Data Understanding

- **Dataset Content:** The dataset contains housing attributes such as location, number of rooms, distance, property type, etc.
- **Input Features (X):** Numerical and categorical housing features.
- **Target (y):** House price.

#### 3. Data Preprocessing

- **Data Cleaning:** Remove or handle missing values.
- **Encoding:** Convert categorical features into numerical using one-hot encoding.
- **Feature Scaling:** Standardize features to ensure fair regularization.

#### 4. Train-Test Split

- **Data Allocation:** 80% training, 20% testing.
- **Reproducibility:** Use `random_state = 42`.

#### 5. Model Selection and Training

- **Lasso Regression (L1):** Adds absolute penalty to coefficients and performs feature selection by shrinking some coefficients to zero.
- **Ridge Regression (L2):** Adds squared penalty to coefficients and reduces their magnitude without eliminating them.

#### 6. Predictions

- **Execution:** Both models predict house prices on test data.
- **Output:** Continuous numerical values.

#### 7. Model Evaluation and Comparison

Performance is evaluated using Mean Squared Error (MSE) and R² Score.

| Model | MSE | R² Score |
|-------|-----|----------|
| Lasso (L1) | 161,700,900,000 | 0.579136 |
| Ridge (L2) | 128,873,000,000 | 0.664578 |

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
|---------|--------|---------|----------|-----------------|
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
