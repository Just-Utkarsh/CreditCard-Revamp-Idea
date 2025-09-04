import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

#from xModelfile.py import rfc_smote, X_test, y_test


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






def strict_model_evaluator(model, X, y, model_name='Model'):
 
    y_pred = model.predict(X)
    try:
        y_prob = model.predict_proba(X)[:,1]
    except AttributeError:
        
        y_prob = np.zeros_like(y_pred)
        warnings.warn("Model does not support predict_proba; ROC AUC won't be informative.")

   
    acc = accuracy_score(y, y_pred)
    pre = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y, y_prob)
    except:
        auc = 0.5  

    
    bias_msg = ""
    if acc > 0.98: bias_msg = "High Score and dataset imbalance detected(accuracy too high for real data). "
    if acc < 0.7:  bias_msg = "Model underfits or performs poorly on this data. "

    
    score = 0
    if acc > 0.92: score += 2
    elif acc > 0.85: score += 1
    if pre > 0.9: score += 2
    elif pre > 0.8: score += 1
    if rec > 0.9: score += 2
    elif rec > 0.8: score += 1
    if f1 > 0.9: score += 2
    elif f1 > 0.8: score += 1
    if auc > 0.93: score += 2
    elif auc > 0.85: score += 1
   
    score = min(score, 10)

    
    comments = []
    if acc < 0.85: comments.append("Low accuracy.")
    if pre < 0.8: comments.append("Model is not precise (many false positives).")
    if rec < 0.8: comments.append("Model misses many true positives (low recall).")
    if abs(pre - rec) > 0.2: comments.append("Model is unbalanced between precision and recall.")
    if auc < 0.8: comments.append("Model has poor ROC AUC (cannot separate classes well).")
    if bias_msg: comments.insert(0, bias_msg)
    if len(comments)==0: comments.append("This model performs very well on provided data, but always verify with out-of-sample data.")

    
    print(f"===== {model_name} EVALUATION REPORT =====")
    print(f"Accuracy:       {acc:.3f}")
    print(f"Precision:      {pre:.3f}")
    print(f"Recall:         {rec:.3f}")
    print(f"F1 Score:       {f1:.3f}")
    print(f"ROC AUC:        {auc:.3f}")
    print(f"Score (strict): {score}/10")
    print("Comments:")
    for c in comments:
        print(" -", c)
    print("="*40)
    return score, comments


strict_model_evaluator(rfc_smote, X_test, y_test, model_name='Random Forest')




