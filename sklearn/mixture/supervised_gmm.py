"""
Supervised Gaussian Mixture Models
"""

import numpy as np
from ..base import ClassifierMixin
from . import GMM, log_multivariate_normal_density


class SupervisedGMM(GMM, ClassifierMixin):
    def fit(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y)
        assert len(X) == len(y)

        self._components = unique_y = np.unique(y)
        self.n_components = n_components = len(unique_y)
        n_samples, n_features = X.shape
        self.n_features = n_features

        ## initialization step
        self.weights_ = np.ones(n_components, dtype=np.float) / n_components
        self.means_ = np.empty((n_components, n_features))
        if self._covariance_type == "spherical":
            self.covars_ = np.empty((n_components, n_features))
        elif self._covariance_type == "diag":
            self.covars_ = np.empty((n_components, n_features))
        elif self._covariance_type == "full":
            self.covars_ = np.empty((n_components, n_features, n_features))
        elif self._covariance_type == "tied":
            self.covars_ = np.empty((n_features, n_features))

        for k, c in enumerate(unique_y):
            mask = (y == c)
            self.weights_[k] = float(mask.sum()) / n_samples
            self.means_[k] = np.mean(X[mask], axis=0)

            if self._covariance_type == "diag":
                self.covars_[k] = np.var(X[mask], axis=0)
            elif self._covariance_type == "spherical":
                self.covars_[k] = np.tile(
                        np.var(X[mask], axis=0).mean(), n_features)
            elif self._covariance_type == "full":
                self.covars_[k] = np.cov(X[mask].T)

        if self._covariance_type == "tied":
            self.covars_ = np.cov(X.T) / n_components

    def predict(self, X):
        idx = GMM.predict(self, X)
        return self._components[idx]

    def precision(self, X, y):
        return ClassifierMixin.score(self, X, y)

    def score(self, X, y):
        s = 0.0
        for k, c in enumerate(self._components):
            idx = (y == c)

            #skip empty component
            if not np.any(idx):
                continue

            if self._covariance_type == "tied":
                s += log_multivariate_normal_density(X[idx],
                    self.means_[k, np.newaxis], self.covars_,
                    self._covariance_type).sum()
            else:
                s += log_multivariate_normal_density(X[idx],
                    self.means_[k, np.newaxis], self.covars_[k, np.newaxis],
                        self._covariance_type).sum()

        return s
