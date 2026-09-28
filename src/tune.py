from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from preprocessing import load_and_prepare_data


X_train,X_test,X_train_scaled,X_test_scaled,y_train,y_test,scaler = load_and_prepare_data()

lr_parameters = {

    "C":[0.001,0.01,0.1,1,10,100],

    "solver":["liblinear","lbfgs"]
    }

dt_parameters={

"max_depth":[3,5,10,20,None],

"criterion":["gini","entropy"],

"min_samples_split":[2,5,10]
}

rf_parameters={

"n_estimators":[100,200,300],

"max_depth":[5,10,20,None],

"max_features":["sqrt","log2"]

}

models = {

"Logistic Regression":( LogisticRegression(),lr_parameters),


"Decision Tree":( DecisionTreeClassifier(), dt_parameters),


"Random Forest":( RandomForestClassifier(), rf_parameters)

}

for name,(model,parameters) in models.items():

    grid = GridSearchCV(
        model,
        parameters,
        cv=5,
        scoring="accuracy"
    )

    grid.fit(X_train,y_train)

    print(name)
    print(grid.best_params_)
    print(grid.best_score_)