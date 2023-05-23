##Data preprocessing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

df = pd.read_csv("train_2.csv", sep=",", index_col=False)
# column names
df.index = df.index + 1
df.columns = [f"col{i}" for i in range(1, len(df.columns)+1)]

# saving new version of csv document
df.to_csv('new_train.csv', index=False)
print(df)

# Delete columns which exceeds threshold limit
threshold = 20
missing_percentages = df.isna().mean() * 100
columns_to_drop = missing_percentages[missing_percentages > threshold].index.tolist()
non_missing_cols = df.columns[df.isna().mean() == 0]
print("columns that are not include missing values:", non_missing_cols)

for col in df.columns:
    if missing_percentages[col] > threshold:
        print(col, "- drop")
    else:
        print(col, "- dont drop")

df.drop(columns_to_drop, axis=1, inplace=True)
print(df.shape)

#description of data types
data_type_counts = df.dtypes.value_counts()
print(data_type_counts)

object_cols = []
float_cols = []
binary_cols = []
label=[]

for col in df.columns:
    if df[col].dtype == "object":
        object_cols.append(col)
    elif df[col].dtype == "float64":
        float_cols.append(col)
    elif df[col].dtype == 'int64' and set(df[col].unique()) == {0, 1}:
        binary_cols.append(col)

#data in the column 122 is all integer and there are no missing values
#we did this for using in machine learning steps in the future
binary_cols.remove("col122")
label.append("col122")


print(f"Object columns: {len(object_cols)}, {object_cols}")
print(f"Float columns: {len(float_cols)}, {float_cols}")
print(f"Integer columns: {len(binary_cols)}, {binary_cols}")
print(f"Label column:", label)

from sklearn.preprocessing import LabelEncoder
# create a label encoder object
le = LabelEncoder()

# looking for binary but not numeric columns
for col in object_cols:
    df[col] = le.fit_transform(df[col])
    if len(df[col].unique()) == 2:
        binary_cols.append(col)
        object_cols.remove(col)
   
print("New integer columns:", len(binary_cols), binary_cols)
print("New object columns:", len(object_cols), object_cols)

#replacement of missing values with numerical values
df[float_cols] = df[float_cols].fillna(df[float_cols].median())
df[binary_cols] = df[binary_cols].fillna(df[binary_cols].mode().iloc[0])
df[object_cols] = df[object_cols].fillna(df[object_cols].mode().iloc[0])

print(df.isnull().sum().sum())

valuecount = df["col122"].value_counts()
percentages = valuecount/len(df["col122"])
print(percentages)

# select only the numerical columns for calculation correlation matrix
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
correlation = df[numeric_cols].corr().round(2)

# select the top and bottom N features based on correlation
N = 20
highest_correlation = correlation.unstack().sort_values(ascending=False)[:N]
lowest_correlation = correlation.unstack().sort_values(ascending=True)[:N]

# combine the top and bottom N features into a single series
top_bottom = pd.concat([highest_correlation, lowest_correlation])

# extract the unique feature names from the top and bottom index
feature_names = list(set(top_bottom.index.get_level_values(0)) | set(top_bottom.index.get_level_values(1)))
correlation_top_bottom = df[feature_names].corr()

# filter out features with correlation less than 0.3
correlation_top_bottom = correlation_top_bottom[(correlation_top_bottom > 0.3) | (correlation_top_bottom < -0.3)]

# second plot which includes more important results and not crowded much as first plot as
sns.set(rc={'figure.figsize':(12,10)})
sns.heatmap(correlation_top_bottom, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True, annot_kws={"size": 10, "color": "black"})
plt.title('Correlation Heatmap')
plt.show()

# Find features with correlation greater than 0.8
high_corr_features = np.where(correlation > 0.8)

# Get unique feature pairs
high_corr_features = [(correlation.columns[x], correlation.columns[y]) for x, y in zip(*high_corr_features) if x != y and x < y]

# Print the high correlation pairs
for feature_pair in high_corr_features:
    print(feature_pair)
     # Check if features contain same data
    if np.array_equal(df[feature_pair[0]], df[feature_pair[1]]):
        print(f"Features {feature_pair[0]} and {feature_pair[1]} contain same data.")
    else:
        print(f"Features {feature_pair[0]} and {feature_pair[1]} do not contain same data.")

# Define columns to check
columns_to_check = ['col10', 'col38', 'col14', 'col67', 'col28', 'col36', 'col29', 'col53', 'col51', 'col72', 'col73', 'col119', 'col74', 'col100']

# Calculate average correlation for each feature
avg_corr = {}
for column in columns_to_check:
    corr = correlation.loc[column, [x for x in correlation.columns if x != column]].mean()
    avg_corr[column] = corr
# Print average correlations
print(avg_corr)

#According to result of average correlations we have to drop col10, col67, col36, col51, col73, col74
df = df.drop(['col10', 'col67', 'col36', 'col51', 'col73', 'col74'], axis=1)

print(df.shape)

float_feats = []
object_feats = []
binary_feats = []
label = []

from scipy.stats import kstest, normaltest

float_feats = []
binary_feats = []
label = []

for col in df.columns:
    if df[col].nunique() == 2:
        binary_feats.append(col)
    elif df[col].nunique() > 30:
        if df[col].dtype == 'float64':
            float_feats.append(col)
        elif df[col].dtype == 'int64':
            binary_feats.append(col)
    else:
        object_feats.append(col)

binary_feats.remove("col122")
label.append("col122")


for col in float_feats:
    stat, p = kstest(df[col], 'norm')
    print(col, "floats p-Test: Statistics={0:.3f}".format(stat, p))
    if p < 0.05:
       print(col, "is not normally distributed")
    else:
        print(col, "is normally distributed")

for col in binary_feats:
    stat, p = normaltest(df[col])
    print(col, "integers p-Test: Statistics={0:.3f}".format(stat, p))
    if p < 0.05:
        print(col,"is not normally distributed")
    else:
        print(col, "is normally distributed")
       
###DENSITY AND HISTOGRAM of float_feats  
plt.figure(figsize=(20, 5))
n_cols = 5
n_rows = int(np.ceil(len(float_feats) / n_cols))

for i, col in enumerate(float_feats):
    plt.subplot(n_rows, n_cols, i+1)
    sns.kdeplot(df[col], shade=True)
    plt.xlabel(col)

plt.tight_layout()
plt.suptitle("FLOAT_FEATS DENSITY PLOTS", fontsize=14)
plt.show()

n_cols = 6  
n_rows = 2  

for i in range(0, len(binary_feats), n_cols * n_rows):
    plt.figure(figsize=(20, 5))
    subset = binary_feats[i:i + n_cols * n_rows]  

    for j, col in enumerate(subset):
        plt.subplot(n_rows, n_cols, j + 1)
        sns.kdeplot(df[col], shade=True)
        plt.xlabel(col)

    plt.tight_layout()
    plt.suptitle(f"Binary_feats Density Plots - Sayfa {i // (n_cols * n_rows) + 1}", fontsize=14)
    plt.show()
X = df.drop('col122', axis=1)
y = df['col122']

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
