from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from preprocessing import load_and_prepare_data

import pickle

X_train_scalled,X_test_scalled,X_train,X_test,y_train,y_test,scaler=load_and_prepare_data()

models={
    "Logistic Regresison":LogisticRegression(max_iter=10000,solver="liblinear"),
    "Decision Tree":DecisionTreeClassifier(random_state=42),
    "Random Forest":RandomForestClassifier(n_estimators=100,random_state=42)
}
result={}
for name,model in models.items():
    if name=="Logistic Regression":
        model.fit(X_train_scalled,y_train)
        y_pred = model.predict(X_test_scalled)
        print(name)
    else:
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        print(name)
    accuracy=accuracy_score(y_test,y_pred)
    result[name]=accuracy
    print(f"acurrary={accuracy}")
    filename=("models/"+name.replace(" ","_")+".pkl")
    with open(filename,"wb") as file:
        pickle.dump(model,file)

print(result)
max_accuracy_model_name = max(result,key=result.get)
best_accuracy = result[max_accuracy_model_name]

print("\nMost accurate Model:")
print(max_accuracy_model_name)
print(f"Accuracy:{best_accuracy}")



