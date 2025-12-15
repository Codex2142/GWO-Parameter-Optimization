from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score

def get_param_bounds():
    return [(1e-12, 1e-6)], 1

def fitness(position, X_train, y_train, X_val, y_val):
    smooth = position[0] * 1e-6
    model = GaussianNB(var_smoothing=smooth)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return f1_score(y_val, preds)

def train_best_model(position, X_train, y_train):
    smooth = position[0] * 1e-6
    model = GaussianNB(var_smoothing=smooth)
    model.fit(X_train, y_train)
    return model
