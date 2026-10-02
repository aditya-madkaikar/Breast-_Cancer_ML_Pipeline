from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

from preprocessing import load_and_prepare_data

import pickle
import os

X_train,X_test,X_train_scalled,X_test_scalled,y_train,y_test,scaler=load_and_prepare_data()

import numpy as np

models={
    "Logistic Regression":LogisticRegression(max_iter=10000,solver="lbfgs",C=0.1),
    "Decision Tree":DecisionTreeClassifier(random_state=42),
    "Random Forest":RandomForestClassifier(n_estimators=100,random_state=42)
}
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
os.makedirs("models",exist_ok=True)
result={}
for name,model in models.items():
    print("\n==============================")
    print(name)
    print("==============================")
    if name=="Logistic Regression":
        cv_model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy")
        model.fit(X_train_scalled,y_train)
        y_pred = model.predict(X_test_scalled)
    else:
        cv_model=model
        cv_scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=5,
                    scoring="accuracy")
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
    test_accuracy=accuracy_score(y_test,y_pred)
    print("Cross-validation scores:")
    print(cv_scores)
    print("Mean CV accuracy:",cv_scores.mean())
    print("CV standard deviation:",cv_scores.std())
    print("Test accuracy:",test_accuracy)
    filename=("models/"+name.replace(" ","_")+".pkl")
    with open(filename,"wb") as file:
        pickle.dump(model,file)



