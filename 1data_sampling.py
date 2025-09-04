import pandas as pd
#to sample the dataset to get the metrics of the datasheet V1-28


data = pd.read_csv('creditcard_2023.csv')
print("Dataset shape:", data.shape)
print("Columns:", data.columns.tolist())

print(data.head())

print(data.info())

print(data['Class'].value_counts())

print(data.describe())

print(data.isnull().sum())