import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from preprocessing import load_and_prepare_data

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

CONFUSION_DIR = os.path.join(
    RESULTS_DIR,
    "confusion_matrices"
)

os.makedirs(
    CONFUSION_DIR,
    exist_ok=True
)

X_train, X_test, X_train_scaled, X_test_scaled,y_train,y_test,scaler=load_and_prepare_data()

model_files={
        "Logistic Regression":os.path.join(MODELS_DIR,"Logistic_Regression.pkl"),
        "Decision Tree":os.path.join(MODELS_DIR,"Decision_Tree.pkl"),
        "Random Forest":os.path.join(MODELS_DIR,"Random_Forest.pkl"),
        "Tuned Logistic Regression":os.path.join(MODELS_DIR,"Tuned_Logistic_Regression.pkl"),
        "Tuned Decision Tree":os.path.join(MODELS_DIR,"Tuned_Decision_Tree.pkl"),
        "Tuned Random Forest":os.path.join(MODELS_DIR,"Tuned_Random_Forest.pkl")
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
    plt.savefig(os.path.join(CONFUSION_DIR,name.replace(" ","_") + "_confusion_matrix.png"),bbox_inches="tight")
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

plt.savefig(os.path.join(RESULTS_DIR,"roc_curve.png"),bbox_inches="tight")

plt.close()

with open(os.path.join(
        MODELS_DIR,"Random_Forest.pkl"),
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

plt.savefig(os.path.join(
        RESULTS_DIR,"feature_importance.png"),bbox_inches="tight")

plt.close()