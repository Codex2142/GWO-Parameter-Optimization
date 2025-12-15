from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

def get_param_bounds():
    return [(1, 20), (2, 20)], 2  # depth, min_samples_split

def fitness(position, X_train, y_train, X_val, y_val):
    depth = int(position[0] * 20) + 1
    split = int(position[1] * 18) + 2

    model = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_split=split,
        random_state=1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return f1_score(y_val, preds)

def train_best_model(position, X_train, y_train):
    depth = int(position[0] * 20) + 1
    split = int(position[1] * 18) + 2
    model = DecisionTreeClassifier(max_depth=depth, min_samples_split=split)
    model.fit(X_train, y_train)
    return model
