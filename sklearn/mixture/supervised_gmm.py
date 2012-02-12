"""
Supervised Gaussian Mixture Models
"""

import numpy as np
from ..base import ClassifierMixin
from . import GMM, log_multivariate_normal_density


class SupervisedGMM(GMM, ClassifierMixin):
    def _init_params(self, n_components, n_features, init_params='wmc'):
        if 'w' in init_params:
            self.weights_ = np.ones(n_components, np.float) / n_components
        if 'm' in init_params:
            self.means_ = np.empty((n_components, n_features))
        if 'c' in init_params:
            if self._covariance_type == "spherical":
                self.covars_ = np.empty((n_components, n_features))
            elif self._covariance_type == "diag":
                self.covars_ = np.empty((n_components, n_features))
            elif self._covariance_type == "full":
                self.covars_ = np.empty((n_components, n_features, n_features))
            elif self._covariance_type == "tied":
                self.covars_ = np.empty((n_features, n_features))
            
    def fit(self, X, y, params='wmc', init_params="wmc"):

        X = np.asarray(X)
        y = np.asarray(y)
        assert len(X) == len(y)

        self._components = unique_y = np.unique(y)
        self.n_components = n_components = len(unique_y)
        n_samples, n_features = X.shape
        self.n_features = n_features

        ## initialization step
        self._init_params(n_components, n_features, init_params)

        for k, c in enumerate(unique_y):
            mask = (y == c)
            if 'w' in params:
                self.weights_[k] = float(mask.sum()) / n_samples
            if 'm' in params:
                self.means_[k] = np.mean(X[mask], axis=0)
            if 'c' in params:
                if self._covariance_type == "diag":
                    self.covars_[k] = np.var(X[mask], axis=0)
                elif self._covariance_type == "spherical":
                    self.covars_[k] = np.tile(
                            np.var(X[mask], axis=0).mean(), n_features)
                elif self._covariance_type == "full":
                    self.covars_[k] = np.cov(X[mask].T)

        if 'c' in params and self._covariance_type == "tied":
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
