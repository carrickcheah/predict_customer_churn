# README.md

## Introduction

The dataset presents three key problems to solve through data mining:

1. **Customer Churn Prediction**: Identify customers most likely to leave, enabling targeted retention strategies to reduce churn and enhance loyalty.
2. **Customer Segmentation**: Segment customers based on attributes like order preferences, payment methods, and app usage to create personalized marketing strategies and improve customer service.
3. **Order Pattern Analysis**: Understand purchasing behaviors and predict future sales trends for optimized inventory management.

Using data mining techniques like classification, clustering, and association rule mining, these insights can drive improved decision-making and customer-centric actions.

## Objectives

- **Predictive Model**: Build a model to identify customers at risk of leaving, enabling proactive retention.
- **Exploratory Analysis**: Provide insights into customer behavior to inform decision-making.

## Problem Understanding Checklist

1. **Problem Addressed**: Predicting customer churn to help the company identify at-risk customers and take action to retain them. Success is measured by model accuracy and reducing churn by 30%.
2. **Ideal Solution**: A predictive model that identifies churn-prone customers, allowing proactive churn reduction.
3. **Solution Characteristics**: A classifier that predicts whether a customer will churn or not.
4. **Data Mining Modeling**: A classifier to categorize customers as "likely to churn" or "not likely to churn."
5. **Challenges**: Identifying the right features, handling missing data, and ensuring the model generalizes well to unseen data.
6. **Data Domain**: A mix of numerical and categorical variables related to customer behavior and demographics.
7. **Solution Domain**: The model classifies customers into two classes: Churn (1) and No Churn (0).

## Data Understanding Checklist

1. **Data Collection**: A single collection with 20 fields (columns) and 5630 entries (rows).
2. **Data Sample**: One customer’s data across 20 different attributes.
3. **Data Size**: 5630 rows and 20 columns, using 879.8+ KB of memory.
4. **Data Storage**: Stored in an Excel file, inferred from the pandas DataFrame.
5. **Baseline Metrics**: Total number of customers (5630) and churn status (Churn column). Missing values in columns like Tenure, WarehouseToHome, etc.
6. **Sampling Duration/Rate**: Not explicitly mentioned; appears to be a one-time historical data collection.
7. **Varying Settings/Conditions**: No explicit information on varying settings or conditions.
8. **Data Source**: Likely customer information from the e-commerce platform, collected via customer activities, transactions, and surveys.
9. **Data Field Meaning**: Each field represents a different attribute of a customer's interaction with the platform.
10. **Possible Values**: Varies by field (e.g., Gender: Male, Female; Churn: 0 (No), 1 (Yes)).
11. **Data Field Collection Purpose**: To understand customer behavior, preferences, and identify churn risks.
12. **Field Relationships**: Some fields are related (e.g., HourSpendOnApp could influence Churn).
13. **Field Completeness**: Several fields have missing values.

## Data Preparation Methods

1. **Handling Missing Values**: Techniques like backfill, forward fill, random sampling, and KNN Imputer were used to minimize data loss while maintaining accuracy.
2. **Encoding Categorical Variables**: Label Encoding was used to convert categorical columns to numbers.
3. **Handling Outliers**: The Interquartile Range (IQR) method was used to remove outliers.
4. **Data Normalization**: MinMaxScaler was applied to scale features between 0 and 1.
5. **Handling Class Imbalance**: SMOTETomek was used to balance the target variable by oversampling the minority class and undersampling the majority class.

## Data Modeling and Evaluation

### Model Training

Six machine learning models were trained:

1. **Logistic Regression**
2. **Support Vector Machine (SVM)**
3. **Decision Tree**
4. **Random Forest**
5. **XGBoost**
6. **AdaBoost**

### Model Performance

- **Logistic Regression**: Moderate accuracy (train: 0.78, test: 0.78).
- **Support Vector Machine**: Improved test accuracy (0.85).
- **Decision Tree**: Perfect training accuracy (1.0), but overfitting.
- **Random Forest**: Strong performance (test: 0.96).
- **XGBoost**: Best performance with high test accuracy (0.97) and minimal overfitting.
- **AdaBoost**: Balanced performance (train: 0.88, test: 0.88).

### Best Model: XGBoost

- **Accuracy**: 97.14%
- **ROC-AUC**: 0.971
- **Precision, Recall, F1-Score**: Consistently around 0.971

### Rationale for Selecting XGBoost

XGBoost was selected as the best model due to its high-test accuracy and excellent balance between training and testing accuracy, suggesting minimal overfitting. It provided the best performance on the test dataset, making it the most reliable for generalization.

## Conclusion

The XGBoost model demonstrated excellent predictive performance, making it the best choice for identifying customers at risk of churn. This model can significantly aid in proactive customer retention strategies, ultimately enhancing customer loyalty and reducing churn.

---

This README.md provides a comprehensive overview of the dataset, objectives, problem understanding, data preparation methods, and model evaluation, making it an attractive and informative document for interview purposes.
