import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
df = pd.read_csv("kidney_disease.csv")
print("First 5 rows:\n", df.head())     
print("\nDataset shape:", df.shape)     
print("\nColumn names:\n", df.columns) 
print("\nData types:\n", df.dtypes)     

