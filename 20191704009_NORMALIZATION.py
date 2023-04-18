import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv("/Users/berketuran/Downloads/train2.csv", sep=";", header=None)
print(df) 

# Veri setini kaydedin
df.to_csv('train_new.csv', index=False)

# Changing title of column and then remove first row which include titles
df.columns = ['PassengerID', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
df = df.iloc[1:]

df.head()  # Show the first few rows of the dataset
df.info()  # Display information about the dataset
df.describe()  # Show statistical summary of the dataset
df.shape


###Normalize "FARE" column
scaler = MinMaxScaler()
new_df['Fare_normalized'] = scaler.fit_transform(df[['Fare']])
print(new_df['Fare_normalized'])
#  Save last version of dataset
new_df.to_csv('titanic_last.csv', index=False)
new_df = pd.read_csv('titanic_last.csv')
print(new_df.head())
# Calculation of family relations
new_df["FamilySize"] = new_df["SibSp"] + new_df["Parch"] + 1
# Alone ones
new_df.loc[new_df["FamilySize"] == 1, "FamilySize"] = 0
# Segment the age variable
new_df['Age'] = new_df['Age'].replace(',', '.').astype(float)
new_df["Age"] = new_df["Age"].astype(float)
bins = [0, 5, 14, 19, 24, 35, 60, 110]
labels = ["Baby", "Child", "Teenager", "Student", "Young Adult", "Adult", "Senior"]
new_df["AgeGroup"] = pd.cut(new_df["Age"], bins=bins, labels=labels)
