###new dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv("/Users/esradirican/Downloads/train2.csv", sep=";", header=None)
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

# There are some misunderstood data which is described as numeric but written as date accidentaly
