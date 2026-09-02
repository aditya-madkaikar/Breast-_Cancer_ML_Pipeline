import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay
)
from preprocessing import load_and_prepare_data

X_train, X_test, X_train_scaled, X_test_scaled,y_train,y_test,scaler=load_and_prepare_data()

model_files={
        "Logistic Regression":"models/Logistic_Regression.pkl",
        "Decision Tree":"models/Decision_Tree.pkl",
        "Random Forest":"models/Random_Forest.pkl"
}

for name,path in model_files.items():
    print(f"evaluating {name}")
    with open (path,"rb") as file:
        model=pickle.load(file)
    if(name=="Logistic Regression"):
        X_eval=X_test_scaled
    else:
        X_eval=X_test
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_eval,
        y_test
    )
    plt.title(name)
    plt.savefig("results/"+ name.replace(" ","_") + "_confusion_matrix.png")
    plt.close()

fig, ax = plt.subplots()

for name, path in model_files.items():

    with open(path, "rb") as file:
        model = pickle.load(file)

    if name == "Logistic Regression":
        X_eval = X_test_scaled
    else:
        X_eval = X_test

    RocCurveDisplay.from_estimator(
        model,
        X_eval,
        y_test,
        name=name,
        ax=ax
    )

plt.title("ROC Curve Comparison")

plt.savefig("results/roc_curve.png")

plt.close()

with open(
    "models/Random_Forest.pkl",
    "rb"
) as file:

    rf_model = pickle.load(file)


importance = rf_model.feature_importances_

feature_names = X_train.columns


indices = importance.argsort()[-10:]


plt.figure(figsize=(8, 6))

plt.barh(
    feature_names[indices],
    importance[indices]
)

plt.xlabel("Feature Importance")

plt.title(
    "Top 10 Random Forest Features"
)

plt.tight_layout()

plt.savefig(
    "results/feature_importance.png"
)

plt.close()