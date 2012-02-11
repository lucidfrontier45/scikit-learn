"""
Semi-supervised Gaussian Mixture Models
"""

import numpy as np
from ..base import ClassifierMixin
from .. import cluster
from .gmm import GMM, _covar_mstep_funcs, log_multivariate_normal_density,\
        distribute_covar_matrix_to_match_covariance_type

class SemisupervisedGMM(GMM, ClassifierMixin):

    def __init__(self, n_components=1, covariance_type='diag',
            random_state=None, thresh=1e-2, min_covar=1e-3, scale_factor=1.0):
        GMM.__init__(self, n_components, covariance_type, random_state,
                thresh, min_covar)
        self._scale_factor = scale_factor

    def fit(self, X_labeled, y, X_unlabeled=None, n_iter=100, thresh=1e-2,
            params='wmc', init_params='wmc'):

        X_labeled = np.asarray(X_labeled)
        y = np.asarray(y)
        X_unlabeled = np.asarray(X_unlabeled)
        
        # validate input data
        assert len(X_labeled) == len(y)
        assert len(X_labeled[0]) == len(X_unlabeled[0])

        self._components =  np.unique(y)
        self.n_components = len(self._components)
        n_samples, n_features = X_labeled.shape
        self.n_features = n_features
        min_covar = self.min_covar

        ## initialization step
        if 'm' in init_params:
            self.means_ = cluster.KMeans(
                k=self.n_components).fit(X_labeled).cluster_centers_
        elif not hasattr(self, 'means'):
            self.means_ = np.zeros((self.n_components, self.n_features))

        if 'w' in init_params or not hasattr(self, 'weigths_'):
            self.weigths_ = np.tile(1.0 / self.n_components, self.n_components)

        if 'c' in init_params:
            cv = np.cov(X_labeled.T)
            if not cv.shape:
                cv.shape = (1, 1)
            self.covars_ = distribute_covar_matrix_to_match_covariance_type(
                cv, self._covariance_type, self.n_components)
        elif not hasattr(self, 'covars'):
            self.covars = distribute_covar_matrix_to_match_covariance_type(
                np.eye(self.n_features), self.covariance_type,
                self.n_components)

        ## learning step

        logprob = []
        for i in xrange(n_iter):
             # Expectation step
            curr_logprob, posteriors = GMM.eval(self, X_unlabeled)
            curr_logprob = self._score_labeled(X_labeled, y) + \
                self._scale_factor * curr_logprob.sum()
            logprob.append(curr_logprob)

            print i, curr_logprob

            # Check for convergence.
            if i > 0 and abs(logprob[-1] - logprob[-2]) < self.thresh:
                self.converged_ = True
                break                

            self._do_mstep(X_labeled, y, X_unlabeled, posteriors,
                            params, min_covar)

        return self

    def _do_mstep(self, X_labeled, y, X_unlabeled, posteriors,
                                                params, min_covar=0):

        n_labeled = len(X_labeled)
        n_samples = n_labeled
        n_unlabeled = len(X_unlabeled)
        n_samples += self._scale_factor * n_unlabeled

        posteriors *= self._scale_factor
        w_unlabeled = posteriors.sum(axis=0)
        avg_obs = np.dot(posteriors.T, X_unlabeled)
        norm = 1.0 / (w_unlabeled[:, np.newaxis] + 10
                * np.finfo(np.float).eps)
        covar_mstep_func = _covar_mstep_funcs[self._covariance_type]
        covars__unlabeled = covar_mstep_func(self, X_unlabeled, posteriors,
                                                avg_obs, norm, min_covar)

        for k, c in enumerate(self._components):
            mask = (y == c)
            w_labeled = float(mask.sum())

            if 'w' in params:
                self.weigths_[k] = (w_labeled + w_unlabeled[k]) / n_samples

            if 'm' in params:
                self.means_[k] = (np.sum(X_labeled[mask], axis=0) \
                    + avg_obs[k]) / (w_labeled + w_unlabeled[k])

            if 'c' in params:
                if self._covariance_type == "diag":
                    self.covars_[k] = np.var(X_labeled[mask], axis=0)
                elif self._covariance_type == "spherical":
                    self.covars_[k] = np.tile(
                            np.var(X_labeled[mask], axis=0).mean(), n_features)
                elif self._covariance_type == "full":
                    self.covars_[k] = np.cov(X_labeled[mask].T)
                
                self.covars_[k] = (self.covars_[k] * w_labeled \
                        + covars__unlabeled[k] * w_unlabeled[k]) \
                        / (w_labeled + w_unlabeled[k])

        if 'c' in params:
            if self._covariance_type == "tied":
                self.covars_ = np.cov(X_labeled.T) / self.n_components
            self.covars_ = (self.covars_ * n_labeled +
                covars__unlabeled * n_unlabeled) / n_samples


    def predict(self, X):
        idx = GMM.predict(self, X)
        return self._components[idx]

    def precision(self, X, y):
        return ClassifierMixin.score(self, X, y)

    def score(self, X_labeled, y, X_unlabeled=None):
        s = 0
        if X_labeled:
            s += self._score_labeled(X_labeled, y)
        if X_unlabeled:
            s += self._score_unlabeled(X_unlabeled)
            
        return s

    def _score_unlabeled(self, X):
        return self._scale_factor * GMM.score(self, X)

    def _score_labeled(self, X, y):
        s = 0.0
        for k, c in enumerate(self._components):
            idx = (y == c)

            #skip empty component
            if not np.any(idx):
                continue

            if self._covariance_type == "tied":
                s += log_multivariate_normal_density(X[idx],
                    self.means_[k, np.newaxis],
                    self.covars_, self._covariance_type).sum()
            else:
                s += log_multivariate_normal_density(X[idx],
                    self.means_[k, np.newaxis],
                    self.covars_[k, np.newaxis], self._covariance_type).sum()
        return s
