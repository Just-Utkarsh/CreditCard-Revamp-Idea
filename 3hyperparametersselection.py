import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('credit_card_fraud_2023.csv') 
data = data.drop(columns=['id'])
X = data.drop(columns=['Class'])
y = data['Class']

# 1. Pearson Correlation: Remove features with correlation higher than 0.9
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
X_reduced_corr = X.drop(columns=to_drop)

print(f"Features dropped by correlation: {to_drop}")
mi = mutual_info_classif(X_reduced_corr, y, discrete_features=False)
mi_series = pd.Series(mi, index=X_reduced_corr.columns)
top_mi_features = mi_series.sort_values(ascending=False).head(15).index.tolist()

print(f"Top features by Mutual Information: {top_mi_features}")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_reduced_corr, y)

importances = rf.feature_importances_
rf_series = pd.Series(importances, index=X_reduced_corr.columns)
top_rf_features = rf_series.sort_values(ascending=False).head(15).index.tolist()

print(f"Top features by Random Forest: {top_rf_features}")
final_features = list(set(top_mi_features) | set(top_rf_features))
print(f"Final selected features count: {len(final_features)}")
print(f"Selected features: {final_features}")
X_final = X_reduced_corr[final_features]
data = pd.read_csv('credit_card_fraud_2023.csv')
data = data.drop(columns=['id'])
X = data[final_features]
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

#hyperparameter for gridsearchcv
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}

#3foldsearch
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2, scoring='f1')

#in training data
grid_search.fit(X_train, y_train)

#final parameter :) mokshit lala on top
print("Best Hyperparameters:", grid_search.best_params_)
best_rf = grid_search.best_estimator_

y_pred = best_rf.predict(X_test)

#perfpormance measure
print("Classification Report:\n", classification_report(y_test, y_pred))

#
#cm = confusion_matrix(y_test, y_pred)
#sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
    #        xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
#plt.xlabel('Predicted')
#plt.ylabel('Actual')
#plt.title('Confusion Matrix')
#plt.show()
#



#final score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
