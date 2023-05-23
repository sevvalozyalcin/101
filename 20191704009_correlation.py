###Correlation
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
