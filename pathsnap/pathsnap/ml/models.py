"""Explicit baseline configurations and fold-local preprocessing."""
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASELINES = {
    "logistic_regression": dict(C=1.0, max_iter=2000, solver="lbfgs"),
    "knn": dict(n_neighbors=15, weights="distance", algorithm="brute", leaf_size=40),
    "decision_tree": dict(max_depth=12, min_samples_leaf=20),
    "random_forest": dict(n_estimators=120, max_depth=18, min_samples_leaf=10, max_features="sqrt"),
    "hist_gradient_boosting": dict(max_iter=120, learning_rate=0.1, max_leaf_nodes=31,
                                   min_samples_leaf=30, l2_regularization=1.0, early_stopping=False),
}


def make_model(name, params=None, seed=42, jobs=4):
    cfg = dict(BASELINES[name])
    cfg.update(params or {})
    constructors = {"logistic_regression": LogisticRegression, "knn": KNeighborsClassifier,
                    "decision_tree": DecisionTreeClassifier, "random_forest": RandomForestClassifier,
                    "hist_gradient_boosting": HistGradientBoostingClassifier}
    if name != "knn":
        cfg["random_state"] = seed
    if name in ("knn", "random_forest"):
        cfg["n_jobs"] = jobs
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if name in ("logistic_regression", "knn"):
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", constructors[name](**cfg)))
    return Pipeline(steps)


def suggest_params(trial, name):
    if name == "logistic_regression":
        return {"C": trial.suggest_float("C", 1e-3, 100, log=True)}
    if name == "knn":
        return {"n_neighbors": trial.suggest_int("n_neighbors", 5, 51, step=2),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "p": trial.suggest_int("p", 1, 2)}
    if name == "decision_tree":
        return {"max_depth": trial.suggest_int("max_depth", 5, 24),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100, log=True),
                "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 1e-3)}
    if name == "random_forest":
        return {"n_estimators": trial.suggest_int("n_estimators", 80, 200, step=40),
                "max_depth": trial.suggest_int("max_depth", 10, 26),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 60, log=True),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.5, 1.0])}
    return {"learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            "max_iter": trial.suggest_int("max_iter", 80, 240, step=40),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63, step=8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 15, 120, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.01, 20, log=True)}
