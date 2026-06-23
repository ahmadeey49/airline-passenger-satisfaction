# Airline Passenger Satisfaction Prediction ✈️

This repository contains an end-to-end Machine Learning pipeline built using **Python** and **Scikit-learn** to predict airline passenger satisfaction and identify the key operational drivers that influence customer loyalty.

## 📌 Project Overview
Customer retention is crucial for commercial airlines. This project analyzes passenger survey data using a **Binomial Logistic Regression** model. By classifying passengers into "Satisfied" or "Neutral/Dissatisfied", the airline can translate statistical outputs into data-backed business insights to improve service quality.

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Visualization:** Matplotlib

## ⚙️ Data Preprocessing & Modeling Workflow
1. **Data Cleaning:** Handled missing values through row-wise deletion (`dropna()`).
2. **Target Encoding:** Encoded the target variable `satisfaction` using `LabelEncoder`.
3. **Categorical Encoding:** Transformed multi-class categorical features using One-Hot Encoding (`pd.get_dummies`) with `drop_first=True` to avoid the dummy variable trap.
4. **Data Splitting:** Applied an 80/20 train-test split, stratified by the target variable to maintain class balance.
5. **Model Training:** Fit a `Logistic Regression` classifier optimized with a maximum of 5,000 iterations to ensure mathematical convergence.

## 📊 Model Evaluation Metrics
The model performance was evaluated using multiple classification metrics on the testing set:
* **Accuracy:** Overall correctness of the model.
* **Precision:** Accuracy of positive (Satisfied) predictions; helps minimize false positives.
* **Recall:** Ability to capture all satisfied passengers; balances the true positive/false positive trade-offs.
* **Confusion Matrix:** Plotted via Matplotlib to inspect exact prediction errors visually.

## 💡 Key Business Insights & Drivers
By extracting and evaluating the **Logistic Regression Coefficients**, the project ranks features based on their absolute statistical impact:
* **Positive Drivers:** Strategic features where investment directly increases the probability of passenger satisfaction.
* **Negative Drivers:** Critical operational bottlenecks or pain points (e.g., delays, poor service) that significantly hurt customer retention.

## 🚀 Business Recommendations
1. **Targeted Investment:** Prioritize resource allocation toward high-coefficient positive service features.
2. **Operational Fixes:** Address and mitigate customer touchpoints linked to major negative coefficients.
3. **Proactive Intervention:** Use model probability scores (`predict_proba`) to flag and target at-risk dissatisfied passengers before they churn.
4. **Continuous Monitoring:** Audit the Precision-Recall trade-off regularly prior to full deployment.

## ⚠️ Model Limitations
* Assumes a strict linear relationship between features and the log-odds of the target variable.
* Model insights are strictly dependent on the representativeness and quality of the historical survey data.
* Future improvements could involve feature engineering or testing non-linear algorithms such as **Random Forests** or **XGBoost**.
*
