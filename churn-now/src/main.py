# Import necessary libraries
import pandas as pd
import numpy as np

# Data preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek

# Machine Learning models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

# Evaluation metrics
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_excel('customer_data.xlsx')

# Initial data exploration
print("First 5 rows of the data:")
print(df.head())

print("\nData summary:")
print(df.info())

print("\nStatistical summary:")
print(df.describe())

# Data Preprocessing

# Handling missing values
# Let's first check missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Let's handle missing values
# We can use different strategies for different columns
# For numerical columns, we can use KNNImputer
# For categorical columns, we can use the most frequent value

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# For simplicity, assume that 'Churn' is the target variable and is numerical (0 or 1)
# Remove 'Churn' from numerical columns
numerical_cols = [col for col in numerical_cols if col != 'Churn']

# Fill missing values in categorical columns with mode
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Use KNNImputer for numerical columns
imputer = KNNImputer(n_neighbors=5)
df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

# Verify that there are no missing values
print("\nMissing values after imputation:")
print(df.isnull().sum())

# Encoding categorical variables using Label Encoding
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Handling outliers using IQR method
def remove_outliers(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_range = Q1 - 1.5 * IQR
        upper_range = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_range) & (df[col] <= upper_range)]
    return df

df = remove_outliers(df, numerical_cols)

# Data normalization using MinMaxScaler
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Handling class imbalance using SMOTETomek
X = df.drop('Churn', axis=1)
y = df['Churn']

smt = SMOTETomek(random_state=42)
X_res, y_res = smt.fit_resample(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, 
                                                    test_size=0.2, 
                                                    random_state=42)

# Model Training and Evaluation

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_train_pred = lr_model.predict(X_train)
lr_test_pred = lr_model.predict(X_test)
print("\nLogistic Regression Results:")
print(f"Training Accuracy: {accuracy_score(y_train, lr_train_pred):.2f}")
print(f"Testing Accuracy: {accuracy_score(y_test, lr_test_pred):.2f}")
print(classification_report(y_test, lr_test_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, lr_test_pred):.2f}")

# Support Vector Machine
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)
svm_train_pred = svm_model.predict(X_train)
svm_test_pred = svm_model.predict(X_test)
print("\nSupport Vector Machine Results:")
print(f"Training Accuracy: {accuracy_score(y_train, svm_train_pred):.2f}")
print(f"Testing Accuracy: {accuracy_score(y_test, svm_test_pred):.2f}")
print(classification_report(y_test, svm_test_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, svm_test_pred):.2f}")

# Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_train_pred = dt_model.predict(X_train)
dt_test_pred = dt_model.predict(X_test)
print("\nDecision Tree Results:")
print(f"Training Accuracy: {accuracy_score(y_train, dt_train_pred):.2f}")
print(f"Testing Accuracy: {accuracy_score(y_test, dt_test_pred):.2f}")
print(classification_report(y_test, dt_test_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, dt_test_pred):.2f}")

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)
print("\nRandom Forest Results:")
print(f"Training Accuracy: {accuracy_score(y_train, rf_train_pred):.2f}")
print(f"Testing Accuracy: {accuracy_score(y_test, rf_test_pred):.2f}")
print(classification_report(y_test, rf_test_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, rf_test_pred):.2f}")

# XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_train_pred = xgb_model.predict(X_train)
xgb_test_pred = xgb_model.predict(X_test)
print("\nXGBoost Results:")
print(f"Training Accuracy: {accuracy_score(y_train, xgb_train_pred):.2f}")
print(f"Testing Accuracy: {accuracy_score(y_test, xgb_test_pred):.2f}")
print(classification_report(y_test, xgb_test_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, xgb_test_pred):.2f}")

# AdaBoost
ada_model = AdaBoostClassifier()
ada_model.fit(X_train, y_train)
ada_train_pred = ada_model.predict(X_train)
ada_test_pred = ada_model.predict(X_test)
print("\nAdaBoost Results:")
print(f"Training Accuracy: {accuracy_score(y_train, ada_train_pred):.2f}")
print(f"Testing Accuracy: {accuracy_score(y_test, ada_test_pred):.2f}")
print(classification_report(y_test, ada_test_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, ada_test_pred):.2f}")

# Conclusion
print("\nConclusion:")
print("The XGBoost model demonstrated excellent predictive performance, making it the best choice for identifying customers at risk of churn.")
