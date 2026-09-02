from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd

def load_and_prepare_data():
    cancer=load_breast_cancer()
    df=pd.DataFrame(cancer.data,columns=cancer.feature_names)
    df["diagnosis"] = pd.Series(cancer.target).map({
    0:1,
    1:0
}) 

    X=df[cancer.feature_names]
    y=df["diagnosis"]

    X_train,X_test,y_train,y_test=train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler=StandardScaler()
    X_train_scalled=scaler.fit_transform(X_train)
    X_test_saclled=scaler.transform(X_test)

    return(
        X_train,
        X_test,
        X_train_scalled,
        X_test_saclled,
        y_train,
        y_test,
        scaler
    )
