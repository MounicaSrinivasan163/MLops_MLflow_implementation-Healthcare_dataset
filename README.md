# Healthcare Data Analysis & Machine Learning Pipeline

## 📌 Project Overview
This project performs **exploratory data analysis (EDA)** and builds **machine learning models** to predict healthcare-related insights using the provided dataset.  
We apply **RandomizedSearchCV** for hyperparameter tuning and **AdaBoost** with different base estimators, tracking results using **MLflow** for experiment management.

---

## 📊 Dataset Description

| Column               | Description |
|----------------------|-------------|
| Name                 | Patient's full name (identifier - not used in modeling) |
| Age                  | Age of the patient in years |
| Gender               | Gender of the patient (Male/Female/Other) |
| Blood Type           | Patient's blood group (e.g., A+, B-, O+) |
| Medical Condition    | Primary diagnosed medical condition |
| Date of Admission    | Date when the patient was admitted |
| Doctor               | Attending doctor’s name |
| Hospital             | Hospital name |
| Insurance Provider   | Insurance company covering the patient |
| Billing Amount       | Total billing amount in USD |
| Room Number          | Assigned hospital room number |
| Admission Type       | Type of admission (Emergency, Elective, etc.) |
| Discharge Date       | Date of patient discharge |
| Medication           | Prescribed medication(s) |
| Test Results         | Summary of lab test results (Normal/Abnormal) |

---

## 🔍 Feature Classification

| Feature              | Type         | Notes |
|----------------------|-------------|-------|
| Age                  | Continuous  | Numeric, used in modeling |
| Billing Amount       | Continuous  | Numeric, target or feature |
| Room Number          | Continuous* | Numeric but treated as categorical ID if needed |
| Gender               | Categorical | Encoded for ML |
| Blood Type           | Categorical | Encoded for ML |
| Medical Condition    | Categorical | Encoded for ML |
| Date of Admission    | Categorical/Datetime | Can be converted to datetime features |
| Doctor               | Categorical | Encoded for ML |
| Hospital             | Categorical | Encoded for ML |
| Insurance Provider   | Categorical | Encoded for ML |
| Admission Type       | Categorical | Encoded for ML |
| Discharge Date       | Categorical/Datetime | Used for length-of-stay calculation |
| Medication           | Categorical | Encoded for ML |
| Test Results         | Categorical | Encoded for ML |

\* `Room Number` is numeric but often treated as an identifier, not a continuous feature.

---

## 💡 Potential Use Cases

- Predicting **billing amounts** based on patient and treatment data.
- Identifying **high-risk patients** from admission details.
- Forecasting **length of stay** for resource planning.
- Detecting **patterns in medical conditions** across hospitals.
- Insurance fraud detection.

---

## 🧠 Machine Learning Approach

- **Algorithm:** AdaBoost (Adaptive Boosting)  
- **Base Estimators Tested:**
  - `DecisionTreeRegressor`
  - `LinearRegression`
  - `SVR`
- **Hyperparameter Tuning:** `RandomizedSearchCV`
  - Randomly samples from the hyperparameter space for efficient search.
- **Performance Metrics:** MSE, MAE, RMSE

---

## 🛠 Tools & Technologies Used

- **Python**
- **Pandas**, **NumPy**
- **Scikit-learn** (ML models & tuning)
- **Matplotlib**, **Seaborn** (visualization)
- **MLflow** (experiment tracking)
- **Jupyter Notebook**

---

## 📦 About MLflow

**MLflow** is an open-source platform to manage the ML lifecycle, including:
- **Tracking**: Log parameters, metrics, and artifacts during model training.
- **Projects**: Package ML code for reproducibility.
- **Models**: Manage and deploy trained models.
- **Registry**: Store and version models.

In this project, MLflow was used to:
- Log training metrics (MSE, MAE, RMSE).
- Compare models with different base estimators.
- Track hyperparameters from RandomizedSearchCV.
- Save the best-performing model for later deployment.

---

## 📂 Folder Structure
