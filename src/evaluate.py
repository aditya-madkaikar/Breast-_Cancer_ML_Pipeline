import pickle
import os

import pandas as pd


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
        "Logistic Regression":"models/Logistic_Regression.pkl",
        "Decision Tree":"models/Decision_Tree.pkl",
        "Random Forest":"models/Random_Forest.pkl"
}

result=[]
for name,path in model_files.items():
    print(f"evaluating {name}")
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
    print(f"false negtive = {fn}")
    print(f"false positive = {fp}")
    print(f"true negtive = {tn}")
    print(f"true positive = {tp}")

    sensitivity=recall_score(y_test,y_pred)
    specificity=tn/(tn+fp)
    f1=f1_score(y_test,y_pred)
    accurary=accuracy_score(y_test,y_pred)
    auc=roc_auc_score(y_test,y_probab)

    result.append({
        "model name":name,
        "Accurary":accurary,
        "sensitivity":sensitivity,
        "f1 value":f1,
        "specificity":specificity,
        "AUC":auc,
        "false negetive":tn,
        "false positive":fp,

    })
df=pd.DataFrame(result)
print(df)

best_model_name=df.sort_values(by=[
    "sensitivity",
    "AUC"
    ],
    ascending=False
).iloc[0]["model name"]

best_model=model_files[best_model_name]
with open("models/best_model.pkl","wb") as file:
    pickle.dump( best_model,file)

df.to_csv(
    "results/model_comparison.csv",
    index=False
)