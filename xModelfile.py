import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


data = pd.read_csv('credit_card_fraud_2023.csv')

if 'id' in data.columns:
    data = data.drop(columns=['id'])

selected_features = ['V11', 'V18', 'V12', 'V7', 'V17', 'V8', 'V3', 'V27',
                     'V21', 'V5', 'V4', 'V16', 'V10', 'V14', 'V1', 'V9', 'V2']
X = data[selected_features]
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

rfc_smote = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rfc_smote.fit(X_train_res, y_train_res)

y_pred = rfc_smote.predict(X_test)
print("==== RFC + SMOTE Performance on Test Set ====")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")


