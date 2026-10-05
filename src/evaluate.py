import pickle
import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODELS_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


from preprocessing import load_and_prepare_data

X_train, X_test, X_train_scaled, X_test_scaled,y_train,y_test,scaler=load_and_prepare_data()

model_files={
        "Logistic Regression":os.path.join(MODELS_DIR,"Logistic_Regression.pkl"),
        "Decision Tree":os.path.join(MODELS_DIR,"Decision_Tree.pkl"),
        "Random Forest":os.path.join(MODELS_DIR,"Random_Forest.pkl"),
        "Tuned Logistic Regression":os.path.join(MODELS_DIR,"Tuned_Logistic_Regression.pkl"),
        "Tuned Decision Tree":os.path.join(MODELS_DIR,"Tuned_Decision_Tree.pkl"),
        "Tuned Random Forest":os.path.join(MODELS_DIR,"Tuned_Random_Forest.pkl")
}

result=[]
for name,path in model_files.items():
    print(f"========{name}==========")
    with open (path,"rb") as file:
        model=pickle.load(file)
    if(name=="Logistic Regression"):
        X_eval=X_test_scaled
    else:
        X_eval=X_test
    y_pred = model.predict(X_eval)
    y_probab = model.predict_proba(X_eval)[:,1]

    cm=confusion_matrix(y_test,y_pred)
    print(cm)
    tn,fp,fn,tp=cm.ravel()
    print(f"false negative = {fn}")
    print(f"false positive = {fp}")
    print(f"true negative = {tn}")
    print(f"true positive = {tp}")

    sensitivity=recall_score(y_test,y_pred)
    specificity=tn/(tn+fp)
    f1=f1_score(y_test,y_pred)
    accurary=accuracy_score(y_test,y_pred)
    auc=roc_auc_score(y_test,y_probab)
    precision = precision_score(y_test,y_pred)

    result.append({
        "model name":name,
        "Accurary":accurary,
        "sensitivity":sensitivity,
        "Precision":precision,
        "f1 value":f1,
        "specificity":specificity,
        "AUC":auc,
        "false negatives":fn,
        "false positives":fp,

    })
df=pd.DataFrame(result)
print("\nMODEL COMPARISON")
print(df)

os.makedirs("results",exist_ok=True)

df.to_csv(os.path.join(RESULTS_DIR,"model_comparison.csv"),
    index=False
)