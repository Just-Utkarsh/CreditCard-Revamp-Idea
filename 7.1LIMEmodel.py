import lime
import lime.lime_tabular
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('credit_card_fraud_2023.csv')
selected_features = ['V11', 'V18', 'V12', 'V7', 'V17', 'V8', 'V3', 'V27',
                     'V21', 'V5', 'V4', 'V16', 'V10', 'V14', 'V1', 'V9', 'V2']
X = data[selected_features]
y = data['Class']
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)
explainer = lime.lime_tabular.LimeTabularExplainer(
    X.values,
    feature_names=selected_features,
    class_names=['Not Fraud', 'Fraud'],
    discretize_continuous=True
)

i = 0
exp = explainer.explain_instance(
    X.values[i],
    rf.predict_proba,
    num_features=5
)

print(exp.as_list())
exp.save_to_file('lime_explanation.html')
