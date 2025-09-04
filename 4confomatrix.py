import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import randint

data = pd.read_csv('credit_card_fraud_2023.csv')
data = data.drop(columns=['id'])

selected_features = ['V11', 'V18', 'V12', 'V7', 'V17', 'V8', 'V3', 'V27',
                     'V21', 'V5', 'V4', 'V16', 'V10', 'V14', 'V1', 'V9', 'V2']
X = data[selected_features]
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

#rf mdoel
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

#smaller parameters per 50k units
param_dist = {
    'n_estimators': randint(50, 150),
    'max_depth': [None, 10, 20],
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt']
}


random_search = RandomizedSearchCV(rf, param_distributions=param_dist,
                                   n_iter=30, scoring='f1', cv=3,
                                   random_state=42, n_jobs=-1, verbose=2)


random_search.fit(X_train, y_train)

print("Best parameters found:", random_search.best_params_)


best_rf = random_search.best_estimator_
y_pred = best_rf.predict(X_test)
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()