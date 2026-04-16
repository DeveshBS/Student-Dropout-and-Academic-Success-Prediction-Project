# Student Dropout & Academic Success Prediction

End‑to‑end ML pipeline to predict student outcomes (Dropout, Enrolled, Graduate) using academic, demographic, and economic features. Built with Scikit‑learn, Pandas, and Matplotlib.

## Dataset

Portuguese higher education dataset (4,424 students, 37 features including grades, enrollment data, unemployment rate, GDP, etc.). Target: 3‑class academic status.

## Models Evaluated

| Model | F1 Score (after tuning) |
|-------|--------------------------|
| Gradient Boosting | **0.865** |
| Random Forest | 0.777 |
| SVM (RBF kernel) | 0.924 (best) |
| Logistic Regression | 0.814 |
| Decision Tree | 0.674 |
| AdaBoost | 0.650 |

> **Best performing:** SVM with RBF kernel (F1 = 0.924) after hyperparameter tuning (GridSearchCV).

## Key Steps

- Data cleaning, label encoding, train/test split (stratified)
- Standardization (StandardScaler) – critical for SVM & logistic regression
- Models: Logistic Regression, SVM (linear/poly/RBF), Decision Tree, Random Forest, AdaBoost, Gradient Boosting
- Hyperparameter tuning with GridSearchCV (5‑fold CV)
- Evaluation: confusion matrix, precision, recall, weighted F1‑score
