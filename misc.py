from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

DATA_URL = "http://lib.stat.cmu.edu/datasets/boston"

def load_data(url: str = DATA_URL) -> pd.DataFrame:
    raw_df = pd.read_csv(url, sep=r"\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    feature_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target
    return df

def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = None):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def preprocess_data(X_train: np.ndarray, X_test: np.ndarray = None):
    """Fit a StandardScaler on X_train and transform train (+ test if provided)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled

def train_model(model, X_train: np.ndarray, y_train: np.ndarray):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> float:
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return mse

def repeated_evaluate(model_factory, X: np.ndarray, y: np.ndarray, runs: int = 5, test_size: float = 0.2, seeds=None):
    """
    model_factory: a callable that returns a fresh untrained model each run (so hyperparams don't leak)
    Returns average MSE across runs and list of individual MSEs.
    """
    if seeds is None:
        seeds = [None]*runs
    mses = []
    for i in range(runs):
        rs = seeds[i]
        X_tr, X_te, y_tr, y_te = split_data(X, y, test_size=test_size, random_state=rs)
        scaler, X_tr_s, X_te_s = preprocess_data(X_tr, X_te)
        model = model_factory()
        model = train_model(model, X_tr_s, y_tr)
        mse = evaluate_model(model, X_te_s, y_te)
        mses.append(mse)
    avg_mse = float(np.mean(mses))
    return avg_mse, mses

def save_model(model, path: str):
    joblib.dump(model, path)