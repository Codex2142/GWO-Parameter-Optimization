from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def get_param_bounds():
    return [(10, 200), (2, 20)], 2

def fitness(position, X_train, y_train, X_val, y_val):
    n = int(position[0] * 190) + 10
    depth = int(position[1] * 18) + 2

    model = RandomForestClassifier(
        n_estimators=n,
        max_depth=depth,
        random_state=1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return f1_score(y_val, preds)

def train_best_model(position, X_train, y_train):
    n = int(position[0] * 190) + 10
    depth = int(position[1] * 18) + 2
    model = RandomForestClassifier(n_estimators=n, max_depth=depth)
    model.fit(X_train, y_train)
    return model
