# 💰 Loan Status Prediction using SVM

This project uses **Support Vector Machine (SVM)** to predict whether a loan application will be approved ✅ or rejected ❌ based on applicant and property-related features. The model helps automate loan decision-making, improving speed and consistency for banks and financial institutions.

---

## 📄 Dataset

- **Source:** [Loan Prediction Dataset](https://www.kaggle.com/datasets/ninzaami/loan-predication) *(or your source link)*
- **Description:** Contains applicant details with the following columns:
  - `Loan_ID`: Unique loan identifier
  - `Gender`: Applicant gender
  - `Married`: Marital status
  - `Dependents`: Number of dependents
  - `Education`: Education level
  - `Self_Employed`: Employment status
  - `ApplicantIncome`: Applicant income
  - `CoapplicantIncome`: Co-applicant income
  - `LoanAmount`: Loan amount requested
  - `Loan_Amount_Term`: Term of loan in months
  - `Credit_History`: Credit history record (1 = good, 0 = bad)
  - `Property_Area`: Area of the property
  - `Loan_Status`: Target (Y = approved, N = rejected)

---

## ⚙️ Technologies Used

- Python
- NumPy
- Pandas
- Seaborn
- Scikit-learn

---

## 🚀 Project Workflow

### 1️⃣ Data Loading & Exploration
- Load the dataset using Pandas.
- Inspect the shape, check for missing values, and explore value distributions.

### 2️⃣ Data Preprocessing
- Handle missing values by dropping or imputing.
- Encode categorical features like Gender, Married, Education, Property Area, etc.
- Convert `3+` dependents to numerical value (e.g., 4).
- Separate features (`X`) and target variable (`Y`).

### 3️⃣ Data Splitting
- Split the dataset into training and test sets (e.g., 90% training, 10% test) using `train_test_split`.

### 4️⃣ Model Training
- Train a **Support Vector Machine (SVM)** classifier with a linear kernel on the training data.

### 5️⃣ Model Evaluation
- Evaluate model performance using **accuracy score** on both training and test data.
- Check for overfitting or underfitting.

### 6️⃣ Prediction System
- Build a simple predictive system to input new applicant data and predict loan approval status instantly.

---

## ✅ Results

- **Training Accuracy:** ~80.4%, showing that the model learns patterns from the data effectively.
- **Test Accuracy:** ~83.3%, indicating good generalization performance on unseen data.


---

## 💡 What We Learned

- How to preprocess real-world tabular data with missing values and categorical columns.
- Encoding categorical variables for machine learning models.
- Training and evaluating a Support Vector Machine for binary classification tasks.
- Creating a basic prediction system to simulate real-life decision support.

---

## 📥 How to Run

1️⃣ **Clone this repository:**

```bash
git clone https://github.com/RONAKBAGRI/Loan-Status-Prediction-using-SVM.git
```

2️⃣ **Install dependencies:**
```bash
pip install numpy pandas seaborn scikit-learn
```

3️⃣ **Run the notebook:**
```bash
jupyter notebook Loan_Status_Prediction_using_SVM.ipynb
```
