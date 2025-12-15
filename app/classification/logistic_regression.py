from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import numpy as np

def get_param_bounds():
    return [(0.001, 100)], 1  # C

def fitness(position, X_train, y_train, X_val, y_val):
    C = position[0] * 100 + 0.001
    model = LogisticRegression(C=C, max_iter=1000, solver="liblinear", random_state=1)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return f1_score(y_val, preds)

def train_best_model(position, X_train, y_train):
    C = position[0] * 100 + 0.001
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    return model
