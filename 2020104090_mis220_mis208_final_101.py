# -*- coding: utf-8 -*-
##MIS220-MIS208 Final Project
import pandas as pd
import numpy as np
df = pd.read_csv("train_2.csv", sep=",", index_col=False)
df.columns = [f"col{i}" for i in range(1, len(df.columns)+1)]
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

binary_cols.remove("col122")
label.append("col122")
print(f"Object columns: {len(object_cols)}, {object_cols}")
print(f"Float columns: {len(float_cols)}, {float_cols}")
print(f"Integer columns: {len(binary_cols)}, {binary_cols}")
print(f"Label column:", label)
