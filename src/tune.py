from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from preprocessing import load_and_prepare_data

import os
import pickle

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODELS_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)


X_train,X_test,X_train_scaled,X_test_scaled,y_train,y_test,scaler = load_and_prepare_data()

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

lr_parameters = {
"model__C":[0.001,0.01,0.1,1,10,100],
"model__solver":["liblinear","lbfgs"]
}

dt_parameters={
"max_depth":[3,5,10,20,None],
"criterion":["gini","entropy"],
"min_samples_split":[2,5,10],
"min_samples_leaf":[1,2,4]
}

rf_parameters={
"n_estimators":[100,200,300],
"max_depth":[5,10,20,None],
"max_features":["sqrt","log2"],
"min_samples_split":[2,5],
"min_samples_leaf":[1,2]
}

models = {
"Logistic Regression":(Pipeline([("scaler", StandardScaler()),
    ("model",LogisticRegression(max_iter=10000))]),lr_parameters),
"Decision Tree":( DecisionTreeClassifier(random_state=42), dt_parameters),
"Random Forest":( RandomForestClassifier(random_state=42), rf_parameters)
}

scoring = {
    "accuracy": "accuracy",
    "sensitivity": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

for name,(model,parameters) in models.items():

    grid = GridSearchCV(
    model,
    parameters,
    cv=cv,
    scoring=scoring,
    refit="roc_auc",
    n_jobs=-1
    )

    grid.fit(X_train,y_train)

    best_model = grid.best_estimator_

    filename = os.path.join(
    MODELS_DIR,
    "Tuned_" + name.replace(" ", "_") + ".pkl"
)
    with open(filename, "wb") as file:
        pickle.dump(best_model, file)
    print("Saved:", filename)

    print("\n==============================")
    print(name)
    print("==============================")
    print(grid.best_params_)
    print(grid.best_score_)
    print("-------EXTRA INFO--------")
    best_index = grid.best_index_

    print("CV ROC-AUC:")
    print(grid.cv_results_["mean_test_roc_auc"][best_index])

    print("CV Sensitivity:")
    print(grid.cv_results_["mean_test_sensitivity"][best_index])

    print("CV Accuracy:")
    print(grid.cv_results_["mean_test_accuracy"][best_index])

    print("CV F1:")
    print(grid.cv_results_["mean_test_f1"][best_index])




    

