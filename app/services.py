import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from .gwo import GrayWolfOptimizer
from .classification import svm
from .config import GWO_CONFIG, RANDOM_STATE

from .classification import (
    svm,
    decision_tree,
    logistic_regression,
    naive_bayes,
    random_forest,
    xgboost
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "cooked")

ALGORITHMS = {
    "svm": svm,
    "decision_tree": decision_tree,
    "logistic_regression": logistic_regression,
    "naive_bayes": naive_bayes,
    "random_forest": random_forest,
    "xgboost": xgboost
}

def run_optimization(algorithm):

    if algorithm not in ALGORITHMS:
        return {"error": "Algorithm not supported"}

    algo = ALGORITHMS[algorithm]

    # Load dataset
    data = pd.read_csv(os.path.join(DATASET_DIR, "train_val.data"))

    X = data.drop("Diagnosis", axis=1).values
    y = data["Diagnosis"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    bounds, dim = algo.get_param_bounds()

    def fitness(position):
        return algo.fitness(position, X_train, y_train, X_val, y_val)

    gwo = GrayWolfOptimizer(
        n_wolves=GWO_CONFIG["n_wolves"],
        max_iter=GWO_CONFIG["max_iter"],
        dim=dim,
        lb=0,
        ub=1
    )

    best_pos, best_score = gwo.optimize(fitness)
    model = algo.train_best_model(best_pos, X_train, y_train)

    # =====================
    # TESTING
    # =====================
    test_data = pd.read_csv(os.path.join(DATASET_DIR, "test.data"))
    X_test = test_data.drop("Diagnosis", axis=1).values
    y_test = test_data["Diagnosis"].values

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = None

    return {
        "algorithm": algorithm,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "auc": auc,
        "best_fitness": best_score
    }

    # Load dataset cooked
    data = pd.read_csv(os.path.join(DATASET_DIR, "train_val.data"))

    X = data.drop("Diagnosis", axis=1).values
    y = data["Diagnosis"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    if algorithm == "svm":
        bounds, dim = svm.get_param_bounds()

        def fitness(position):
            return svm.fitness(position, X_train, y_train, X_val, y_val)

        gwo = GrayWolfOptimizer(
            n_wolves=GWO_CONFIG["n_wolves"],
            max_iter=GWO_CONFIG["max_iter"],
            dim=dim,
            lb=0,
            ub=1
        )

        best_pos, best_score = gwo.optimize(fitness)
        model = svm.train_best_model(best_pos, X_train, y_train)

    else:
        return {"error": "Algorithm not supported"}

    # Evaluate on test set
    test_data = pd.read_csv(os.path.join(DATASET_DIR, "test.data"))
    X_test = test_data.drop("Diagnosis", axis=1).values
    y_test = test_data["Diagnosis"].values

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "algorithm": algorithm,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
        "best_fitness": best_score
    }
