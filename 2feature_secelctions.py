import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

data = pd.read_csv('credit_card_fraud_2023.csv') 

data = data.drop(columns=['id'])

X = data.drop(columns=['Class'])
y = data['Class']

#remove combi-rel >0.9
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
X_reduced_corr = X.drop(columns=to_drop)

print(f"Features dropped by correlation: {to_drop}")

#important information collections 
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


#final feature combination and selection for further metrics
X_final = X_reduced_corr[final_features]