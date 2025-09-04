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

rf = RandomForestClassifier(
    n_estimators=138,
    max_depth=None,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=7,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)


y_pred = rf.predict(X_test)
print("Classification Report on Test Set:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

fraud_samples = X_test.loc[y_test == 1].copy()

np.random.seed(42)
noise = np.random.normal(0, 0.01, fraud_samples.shape)
adversarial_samples = fraud_samples + noise

adv_preds = rf.predict(adversarial_samples)

print(f"Original fraud samples count: {fraud_samples.shape[0]}")
print(f"Adversarial frauds detected (predicted=1): {(adv_preds == 1).sum()}")
print(f"Adversarial frauds missed (predicted=0): {(adv_preds == 0).sum()}")