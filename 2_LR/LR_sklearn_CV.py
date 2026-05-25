import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_val_score

from linearna_reg import learn, predict


class LinearRegressionCustom(BaseEstimator, RegressorMixin):

    def __init__(self, lam=0.0, alpha=0.01, max_iter=10000, tol=0.01, method='batch', auto_lr=False):
        self.lam = lam
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        self.auto_lr = auto_lr

    def fit(self, X, y):
        data = pd.DataFrame(X).copy()
        data['_target'] = y
        self.model_ = learn(data, '_target', lam=self.lam, alpha=self.alpha,
                            max_iter=self.max_iter, tol=self.tol,
                            method=self.method, auto_lr=self.auto_lr, verbose=False)
        return self

    def predict(self, X):
        return predict(self.model_, pd.DataFrame(X))


# KORISCENJE

data = pd.read_csv('../data/boston.csv')
target = data.columns[-1]

X = data.drop(columns=[target])
y = data[target]

clf = LinearRegressionCustom(lam=0.1, alpha=0.01, method='batch', auto_lr=True)
scores = cross_val_score(clf, X, y, cv=10, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
print(f"Nas LR - RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std():.4f})")