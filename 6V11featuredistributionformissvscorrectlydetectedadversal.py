import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('credit_card_fraud_2023.csv')
data = data.drop(columns=['id'])

selected_features = ['V11', 'V18', 'V12', 'V7', 'V17', 'V8', 'V3', 'V27',
                     'V21', 'V5', 'V4', 'V16', 'V10', 'V14', 'V1', 'V9', 'V2']
X = data[selected_features]
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("Classification report on test set:")
print(classification_report(y_test, y_pred))

fraud_samples = X_test.loc[y_test == 1].copy()

np.random.seed(42)
noise = np.random.normal(0, 0.01, fraud_samples.shape)
adversarial_samples = fraud_samples + noise

adv_preds = rf.predict(adversarial_samples)


missed_idx = np.where(adv_preds == 0)[0]
missed_samples = fraud_samples.iloc[missed_idx]

print(f"Number of missed adversarial fraud samples: {missed_samples.shape[0]}")

detected_idx = np.where(adv_preds == 1)[0]
detected_samples = fraud_samples.iloc[detected_idx]

features_to_compare = ['V11', 'V18', 'V12', 'V7', 'V17']

for feature in features_to_compare:
    plt.figure(figsize=(8,4))
    plt.hist(missed_samples[feature], bins=30, alpha=0.7, label='Missed')
    plt.hist(detected_samples[feature], bins=30, alpha=0.7, label='Detected')
    plt.title(f'Feature: {feature} distribution in missed vs detected adversarial frauds')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.legend()
    plt.show()