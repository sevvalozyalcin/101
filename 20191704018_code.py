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
