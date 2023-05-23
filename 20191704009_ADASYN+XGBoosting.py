
import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# Choose numeric columns
numeric_cols = X.select_dtypes(include=['float', 'int']).columns
X = X[numeric_cols]

# ADASYN oversampling
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X, y)

# New dataset after Oversampling
df_resampled = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled, columns=['target'])], axis=1)
df_resampled.to_csv('train_oversampled.csv', index=False)

print("Oversampled Veri Seti:")
print(df_resampled)

newvaluecount = df_resampled["col105"].value_counts()
newpercentages = newvaluecount/len(df_resampled["col105"])
print(newpercentages)
df_resampled.shape

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# XGBoost data structure
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# XGBoost parameters
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}
#Training the model
model = xgb.train(params, dtrain, num_boost_round=100)

# Prediction on test dataset
y_pred_proba = model.predict(dtest)
y_pred_binary = [1 if proba >= 0.5 else 0 for proba in y_pred_proba]

# Performance metrics
accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)
auc_roc = roc_auc_score(y_test, y_pred_proba)
cm = confusion_matrix(y_test, y_pred_binary)

# True Positive (TP), False Positive (FP), True Negative (TN), False Negative (FN) values
tp = cm[1, 1]
fp = cm[0, 1]
tn = cm[0, 0]
fn = cm[1, 0]

# RESULTS
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("AUC-ROC:", auc_roc)
print("Confusion Matrix:", cm)
print("True Positive (TP):", tp)
print("False Positive (FP):", fp)
print("True Negative (TN):", tn)
print("False Negative (FN):", fn)
