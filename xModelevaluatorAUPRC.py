import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve, classification_report, confusion_matrix
import numpy as np

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

y_scores = rfc_smote.predict_proba(X_test)[:, 1]

auprc = average_precision_score(y_test, y_scores)

print(f"AUPRC (Average Precision): {auprc:.4f}")

y_pred = (y_scores >= 0.5).astype(int)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


def auprc_model_evaluator(model, X, y, threshold=0.5, model_name='Model'):
    y_scores = model.predict_proba(X)[:,1]
    y_pred = (y_scores >= threshold).astype(int)

    auprc = average_precision_score(y, y_scores)
    acc = np.mean(y_pred == y)
    pre = np.sum((y_pred & y)) / np.sum(y_pred) if np.sum(y_pred) > 0 else 0
    rec = np.sum((y_pred & y)) / np.sum(y) if np.sum(y) > 0 else 0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0

    score = 0
    if auprc > 0.92: score += 4
    elif auprc > 0.85: score += 3
    elif auprc > 0.7: score += 2
    elif auprc > 0.5: score += 1

    if acc > 0.9: score += 2
    elif acc > 0.8: score += 1

    if f1 > 0.9: score += 2
    elif f1 > 0.8: score += 1

    score = min(score, 10)

    comments = []
    if auprc < 0.7: comments.append("Low AUPRC, model has poor precision-recall tradeoff.")
    if acc < 0.8: comments.append("Low accuracy, model may struggle to classify correctly.")
    if f1 < 0.8: comments.append("Low F1 score, poor balance between precision and recall.")
    if len(comments) == 0:
        comments.append("Model performs well on AUPRC and classification metrics.")

    print(f"===== {model_name} AUPRC-Based Evaluation =====")
    print(f"AUPRC (Average Precision): {auprc:.4f}")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {pre:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"Strict Score (out of 10): {score}")
    print("Comments:")
    for comment in comments:
        print(f" - {comment}")
    print("="*40)

    return score, comments


auprc_model_evaluator(rfc_smote, X_test, y_test, model_name='RFC + SMOTE AUPRC Model')