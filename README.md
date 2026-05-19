# 🏦 Task 2: Credit Risk Prediction
### DevelopersHub Data Science Internship

---

## 📌 Objective

Predict whether a loan applicant is likely to default on a loan using machine learning. The goal is to build a binary classification model that helps financial institutions assess credit risk based on applicant information such as income, education, credit history, and property details.

---

## 📂 Dataset

- **Source:** Loan Prediction Dataset — [Kaggle](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- **Training set:** 614 records, 13 features
- **Test set:** 367 records, 12 features
- **Target variable:** `Loan_Status` (Y = Approved, N = Rejected)

**Features include:**
| Feature | Description |
|---|---|
| Gender | Male / Female |
| Married | Applicant marital status |
| Dependents | Number of dependents |
| Education | Graduate / Not Graduate |
| Self_Employed | Self-employment status |
| ApplicantIncome | Monthly income of applicant |
| CoapplicantIncome | Monthly income of co-applicant |
| LoanAmount | Loan amount requested (in thousands) |
| Loan_Amount_Term | Term of loan in months |
| Credit_History | Credit history meets guidelines (1/0) |
| Property_Area | Urban / Semiurban / Rural |

---

## 🔧 Approach

### 1. Data Loading & Exploration
- Loaded training and test CSVs from the Kaggle dataset zip
- Inspected shape, data types, and null value counts
- Identified missing values in: `Gender`, `Married`, `Dependents`, `Self_Employed`, `LoanAmount`, `Loan_Amount_Term`, and `Credit_History`

### 2. Handling Missing Values
- **Numerical features** (`LoanAmount`, `Loan_Amount_Term`): filled with **median** to handle skewed distributions
- **Categorical features** (`Gender`, `Married`, `Dependents`, `Self_Employed`, `Credit_History`): filled with **mode** (most frequent value)

### 3. Exploratory Data Analysis (EDA)
Visualized key relationships between features and loan approval:
- Distribution of **Loan Amount** — right-skewed, majority of loans under ₹200K
- **Applicant Income** vs. Loan Status — higher income does not always guarantee approval
- **Education** impact — graduates had a higher approval rate
- **Credit History** — the single most decisive feature; applicants with good credit history were approved at a significantly higher rate
- **Property Area** — Semiurban applicants had the highest approval rates

### 4. Data Preprocessing
- **Label Encoding:** Binary categorical features (`Gender`, `Married`, `Self_Employed`) mapped to 0/1
- **Ordinal Encoding:** `Dependents` — `3+` converted to `3` then cast to integer
- **One-Hot Encoding:** `Education` and `Property_Area` using `pd.get_dummies(drop_first=True)`
- **Dropped:** `Loan_ID` (identifier, not a feature)
- **Train/Test Split:** 80/20 split → 491 training samples, 123 test samples

### 5. Model Training
- **Algorithm:** Logistic Regression (`sklearn.linear_model.LogisticRegression`)
- Chosen for its interpretability and strong baseline performance on binary classification tasks
- Trained on the preprocessed feature matrix with the encoded target variable `Loan_Status`

### 6. Model Evaluation
- Evaluated on the held-out test set using:
  - **Accuracy Score**
  - **Classification Report** (Precision, Recall, F1-Score per class)
  - **Confusion Matrix**

---

## 📊 Results

| Metric | Value |
|---|---|
| **Accuracy** | **78.86%** |
| Precision — Approved (1) | 76% |
| Recall — Approved (1) | 99% |
| F1-Score — Approved (1) | 86% |
| Precision — Rejected (0) | 95% |
| Recall — Rejected (0) | 42% |
| F1-Score — Rejected (0) | 58% |

---

## 💡 Key Insights

1. **Credit History dominates predictions.** Applicants with a positive credit history were approved at a much higher rate than those without. It was the single most influential feature in the model.

2. **Class imbalance affects recall on rejections.** The dataset has more approved loans than rejected ones, which causes the model to excel at predicting approvals (99% recall) but struggle with identifying true defaulters (42% recall). Techniques like SMOTE or class weighting could improve this in future iterations.

3. **Income alone is not a strong predictor.** High applicant income does not guarantee approval — credit history and property area carry more weight in the model's decisions.

4. **Graduates receive more approvals**, but this is likely correlated with higher income and better credit history rather than education being a direct causal factor.

5. **Semiurban property area** correlates with the highest approval rates, possibly reflecting more stable real estate markets compared to rural areas.

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- **Environment:** Google Colab

---

## 🚀 Future Improvements

- Address class imbalance using SMOTE or class weights
- Try ensemble models (Random Forest, XGBoost) for improved recall on rejections
- Perform feature importance analysis
- Hyperparameter tuning with GridSearchCV
- Cross-validation for more robust evaluation

---

*Part of the DevelopersHub Data Science Internship — Task 2 of ongoing project series.*
