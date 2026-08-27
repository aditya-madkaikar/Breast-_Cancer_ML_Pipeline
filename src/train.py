from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from preprocessing import load_and_prepare_data

import pickle

X_train_scalled,X_test_scalled,X_train,X_test,y_train,y_test,scaler=load_and_prepare_data()

models={
    "logistic regresison":LogisticRegression(max_iter=10000,solver="liblinear"),
    "Decison tree":DecisionTreeClassifier(random_state=42),
    "Random forest":RandomForestClassifier(n_estimators=100,random_state=42)
}
result={}
for name,model in models.items():
    if name=="Logistic regression":
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

print(result)
best_model_name = max(result,key=result.get)
best_accuracy = result[best_model_name]
best_model = models[best_model_name]
print("\nBest Model:")
print(best_model_name)
print(f"Accuracy:{best_accuracy}")

with open("models/best_model.pkl","wb") as file:
    pickle.dump(best_model,file)

