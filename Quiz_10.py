import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix

filename = "./data/09_irisdata.csv"

column_names = ['sepal=length','sepal-width','petal-length','petal-width','class']

data = pd.read_csv(filename, names=column_names)

print("Dataset Shape:", data.shape)
print("\nDataset Summary:")
print(data.describe())
print("\nClass Distribution:")
print(data.groupby('class').size())

scatter_matrix(data, alpha=0.8, figsize=(10, 10))
plt.savefig("./scatter_matrix.png")
plt.close()

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

model = DecisionTreeClassifier()

kfold = KFold(n_splits=10, random_state=5, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print(results.mean())

