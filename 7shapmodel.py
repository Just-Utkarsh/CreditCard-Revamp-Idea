
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv('credit_card_fraud_2023.csv')

if 'id' in data.columns:
    data = data.drop(columns=['id'])

selected_features = ['V11', 'V18', 'V12', 'V7', 'V17', 'V8', 'V3', 'V27',
                     'V21', 'V5', 'V4', 'V16', 'V10', 'V14', 'V1', 'V9', 'V2']

X = data[selected_features]
y = data['Class'] 

rf = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
rf.fit(X, y)

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values[1], X, feature_names=selected_features)

shap.force_plot(
    explainer.expected_value[1],
    shap_values[1][0],
    X.iloc[0],
    matplotlib=True
)
plt.show()

