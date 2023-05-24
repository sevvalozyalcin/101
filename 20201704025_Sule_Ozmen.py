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
