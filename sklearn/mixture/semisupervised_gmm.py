"""
Semi-supervised Gaussian Mixture Models
"""

import numpy as np
from .gmm import GMM
from .supervised_gmm import SupervisedGMM

class SemisupervisedGMM(SupervisedGMM):

    def __init__(self, n_components=1, covariance_type='diag',
            random_state=None, thresh=1e-2, min_covar=1e-3, scale_factor=1.0):
        SupervisedGMM.__init__(self, n_components, covariance_type,
                              random_state, thresh, min_covar)
        self._scale_factor = scale_factor
        self._unsupervised_gmm = GMM(n_components, covariance_type,
                                     random_state, thresh, min_covar)

    def fit(self, X_labeled, y, X_unlabeled=None, n_iter=100,
                params='wmc', init_params='wmc'):

        X_labeled = np.asarray(X_labeled)
        y = np.asarray(y)
        X_unlabeled = np.asarray(X_unlabeled)
        n_labeled = len(X_labeled)
        n_samples = n_labeled
        n_unlabeled = self._scale_factor * len(X_unlabeled)
        n_samples += n_unlabeled

        # validate input data
        assert n_labeled == len(y)
        assert len(X_labeled[0]) == len(X_unlabeled[0])

        self._components =  np.unique(y)
        self.n_components = len(self._components)
        min_covar = self.min_covar
        self._unsupervised_gmm.n_components = self.n_components
        learning_ratio = np.array([n_labeled, n_unlabeled]) / float(n_samples)

        # first initialize by supervisedGMM
        if init_params :
            SupervisedGMM.fit(self, X_labeled, y, params, init_params)

        # set parameters of unsupervised gmm
        self._unsupervised_gmm.weights_ = self.weights_
        self._unsupervised_gmm.means_ = self.means_
        self._unsupervised_gmm.covars_ = self.covars_

        ## learning step
        log_likelihood = [-1e100]
        for i in xrange(n_iter):

            # Expectation step
            unsupervised_log_likelihood, responsibilities =\
                self._unsupervised_gmm.eval(X_unlabeled)
            curr_log_likelihood = self._score_labeled(X_labeled, y) +\
                self._scale_factor * unsupervised_log_likelihood.sum()
            log_likelihood.append(curr_log_likelihood)
            
            # Check for convergence.
            delta_log_likelihood = log_likelihood[-1] - log_likelihood[-2]
            if delta_log_likelihood < 0:
                print "warning, delta_log_likelihood %f > 0" \
                    % delta_log_likelihood
            print i, delta_log_likelihood, self.thresh
            if abs(delta_log_likelihood) < self.thresh:
                self.converged_ = True
                break

            # do unsupervised learning
            self._unsupervised_gmm._do_mstep(X_unlabeled, responsibilities,
                            params, min_covar)

            # do supervised learning
            SupervisedGMM.fit(self, X_labeled, y, params, "")

            # mearge two model
            self._do_merge(learning_ratio, params)

        return self

    def _do_merge(self, learning_ratio, params='wmc'):
        if 'w' in params:
            self.weights_ = self.weights_ * learning_ratio[0] +\
                self._unsupervised_gmm.weights_ * learning_ratio[1]
        if 'm' in params:
            self.means_ = self.means_ * learning_ratio[0] +\
                self._unsupervised_gmm.means_ * learning_ratio[1]
        if 'c' in params:
            self.covars_ = self.covars_ * learning_ratio[0] +\
                self._unsupervised_gmm.covars_ * learning_ratio[1]

        # set parameters of unsupervised gmm
        self._unsupervised_gmm.weights_ = self.weights_
        self._unsupervised_gmm.means_ = self.means_
        self._unsupervised_gmm.covars_ = self.covars_

    def score(self, X_labeled, y, X_unlabeled=None):
        s = 0
        if len(X_labeled) > 0:
            s += self._score_labeled(X_labeled, y)
        if len(X_unlabeled) > 0:
            s += self._score_unlabeled(X_unlabeled)
        return s

    def _score_unlabeled(self, X):
        return self._scale_factor * self._unsupervised_gmm.score(X).sum()

    def _score_labeled(self, X, y):
        return SupervisedGMM.score(self, X, y)