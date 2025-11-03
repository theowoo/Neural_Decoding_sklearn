import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted


class WienerCascade(RegressorMixin, BaseEstimator):
    """sklearn wrapper for numpy polyfit."""

    def __init__(self, degree=3):
        self.degree = degree

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.regr_ = LinearRegression()
        self.regr_.fit(X, y)

        y_pred = self.regr_.predict(X)

        self.p_ = np.polyfit(y_pred, y, self.degree)
        self.is_fitted_ = True

        return self

    def predict(self, X, y=None):

        check_is_fitted(self)
        y_pred = self.regr_.predict(X)
        y_pred = np.polyval(self.p_, y_pred)

        return y_pred
