import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score


def get_param_bounds():
    # C, gamma
    dim = 2
    bounds = [(0, 1), (0, 1)]
    return bounds, dim


def _decode_position(position):
    C = 0.1 + position[0] * (100 - 0.1)
    gamma = 0.0001 + position[1] * (1 - 0.0001)
    return C, gamma


def fitness(position, X_train, y_train, X_val, y_val):
    C, gamma = _decode_position(position)

    model = SVC(
        C=C,
        gamma=gamma,
        kernel="rbf",
        probability=True,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    return f1_score(y_val, y_pred)


def train_best_model(best_position, X_train, y_train):
    C, gamma = _decode_position(best_position)

    model = SVC(
        C=C,
        gamma=gamma,
        kernel="rbf",
        probability=True,
        random_state=1
    )

    model.fit(X_train, y_train)
    return model
