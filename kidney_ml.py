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

#  Data Cleaning 

# Drop id column 
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

# Replace with missing values
df.replace('?', pd.NA, inplace=True)

# Clean target column, removes extra spaces and tab characters
df['classification'] = df['classification'].str.strip()

# Numeric columns stored as object to float
numeric_columns = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 
                   'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Fill missing numeric values
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

# Fill missing categorical values
categorical_columns = df.select_dtypes(include='object').columns.drop('classification')
df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

# Mapping text values
for col in categorical_columns:
    df[col] = df[col].map({
        'yes':1, 'no':0,
        'present':1, 'notpresent':0,
        'good':1, 'poor':0,
        'normal':1, 'abnormal':0
    })

# Encode target column
df['classification'] = df['classification'].map({'ckd':1, 'notckd':0})

print("\nData cleaning completed. Sample data:\n", df.head())

# Prepare Features and Target

X = df.drop('classification', axis=1)  # Features
y = df['classification']               # Target

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Train RF Model

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  
rf_model.fit(X_train, y_train)  # Train model
y_pred = rf_model.predict(X_test)  # Predict on test set

print("\nFirst 10 predictions:", y_pred[:10])

# Evaluate Model

accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
print("\nAccuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)  # Confusion matrix
print("\nConfusion Matrix:\n", cm)

# Feature importance
importances = rf_model.feature_importances_
feature_names = X.columns
print("\nFeature Importances:")
for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.3f}")