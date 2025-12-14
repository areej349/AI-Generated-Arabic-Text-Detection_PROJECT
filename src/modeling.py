

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)



DATA_PATH = "data/processed/"
MODEL_PATH = "models/"
REPORT_PATH = "reports/"

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(REPORT_PATH, exist_ok=True)

FEATURES = [
    "F3_digits_ratio",
    "F24_unique_punct_ratio",
    "F45_adjectives",
    "F66_genitives",
    "F87_gini"
]



def load_data():
    train = pd.read_csv(DATA_PATH + "train_features.csv")
    val   = pd.read_csv(DATA_PATH + "val_features.csv")
    test  = pd.read_csv(DATA_PATH + "test_features.csv")
    return train, val, test



def prepare_features(train, val, test):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(train[FEATURES])
    X_val   = scaler.transform(val[FEATURES])
    X_test  = scaler.transform(test[FEATURES])

    joblib.dump(scaler, MODEL_PATH + "scaler.pkl")

    return (
        X_train, train["label"].values,
        X_val,   val["label"].values,
        X_test,  test["label"].values
    )



def evaluate_model(name, model, X, y):
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else preds

    return {
        "Model": name,
        "Accuracy": accuracy_score(y, preds),
        "Precision": precision_score(y, preds),
        "Recall": recall_score(y, preds),
        "F1-Score": f1_score(y, preds),
        "ROC-AUC": roc_auc_score(y, proba)
    }



def train_baselines(X_train, y_train, X_val, y_val):
    results = []

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    joblib.dump(lr, MODEL_PATH + "logistic_regression.pkl")
    results.append(evaluate_model("Logistic Regression", lr, X_val, y_val))

    nb = GaussianNB()
    nb.fit(X_train, y_train)
    joblib.dump(nb, MODEL_PATH + "naive_bayes.pkl")
    results.append(evaluate_model("Naive Bayes", nb, X_val, y_val))

    return results


def train_random_forest(X_train, y_train, X_val, y_val):
    params = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20]
    }

    rf = RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    )

    grid = GridSearchCV(rf, params, cv=3, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_rf = grid.best_estimator_
    joblib.dump(best_rf, MODEL_PATH + "random_forest.pkl")

    return evaluate_model("Random Forest", best_rf, X_val, y_val)



def train_xgboost(X_train, y_train, X_val, y_val):
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH + "xgboost.pkl")

    return evaluate_model("XGBoost", model, X_val, y_val)



def run_phase4():
    train, val, test = load_data()

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_features(
        train, val, test
    )

    results =
    results.extend(train_baselines(X_train, y_train, X_val, y_val))
    results.append(train_random_forest(X_train, y_train, X_val, y_val))
    results.append(train_xgboost(X_train, y_train, X_val, y_val))

    df_results = pd.DataFrame(results)
    df_results.to_csv(REPORT_PATH + "phase4_results.csv", index=False)

    print("\n Phase 4 Results:")
    print(df_results.sort_values("F1-Score", ascending=False))

    return df_results


if __name__ == "__main__":
    run_phase4()
