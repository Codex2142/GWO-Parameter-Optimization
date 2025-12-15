import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score


# -------------------------------------------------
# Parameter bounds (disamakan gaya dengan SVM)
# -------------------------------------------------
def get_param_bounds():
    # n_estimators, max_depth, learning_rate
    dim = 3
    bounds = [(0, 1), (0, 1), (0, 1)]
    return bounds, dim


# -------------------------------------------------
# Decode posisi GWO → parameter XGBoost
# -------------------------------------------------
def _decode_position(position):
    n_estimators = int(50 + position[0] * (300 - 50))
    max_depth = int(3 + position[1] * (10 - 3))
    learning_rate = 0.01 + position[2] * (0.3 - 0.01)

    return n_estimators, max_depth, learning_rate


# -------------------------------------------------
# Fitness function (F1-score validation)
# -------------------------------------------------
def fitness(position, X_train, y_train, X_val, y_val):
    n_estimators, max_depth, learning_rate = _decode_position(position)

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    return f1_score(y_val, y_pred)


# -------------------------------------------------
# Train model terbaik dari posisi terbaik GWO
# -------------------------------------------------
def train_best_model(best_position, X_train, y_train):
    n_estimators, max_depth, learning_rate = _decode_position(best_position)

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)
    return model
