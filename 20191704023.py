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
