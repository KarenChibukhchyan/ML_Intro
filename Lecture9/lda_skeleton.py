import numpy as np
from scipy import linalg


class My_LDA(object):

    def __init__(self, K):
        self.K = K
        self.explained_variance_ = np.array([])
        self.explained_variance_ratio_ = np.array([])
        self.eigenvectors_ = np.array([])

    def compute_scatters(self, X, y):
        """
        param X: numpy array of shape (M,N)
        param y: numpy array of shape (M), shows to which class each row of X belongs
        return S_w, S_b: scatter within and scatter between matrices which are of shape (N,N)
        """
        N = X.shape[1]
        clusters = np.unique(y)
        S_w = np.zeros(shape=(N, N))
        S_b = np.zeros(shape=(N, N))
        total_mean = np.mean(X, axis=0)
        for c in clusters:
            mask = y == c
            X_c = X[mask]
            mean_c = X_c.mean(axis=0)
            X_c = X_c - mean_c
            S_w += np.dot(X_c.T, X_c)

            N_c = len(X_c)
            mean_c -= total_mean
            mean_c = np.reshape(mean_c, newshape=(-1, len(mean_c)))
            S_b += N_c * np.dot(mean_c.T, mean_c)
        return S_w, S_b

    def fit(self, X, y):
        """
        param X: numpy array of shape (M,N)
        param y: numpy array of shape (M), shows to which class each row of X belongs
        """
        """TODO fit the model,(compute scatter matrices and compute 
        eigenvalues and eigenvectors of S_w^{-1}S_b) and update values of 
        self.explained_variance_, self.explained_variance_ratio_, self.eigenvectors"""
        S_w, S_b = self.compute_scatters(X, y)
        S_w_inverse = np.linalg.inv(S_w)
        self.explained_variance_, self.eigenvectors_ = linalg.eig(np.dot(S_w_inverse, S_b), left=True, right=False)
        sum_ = sum(np.abs(self.explained_variance_ ** 2))
        self.explained_variance_ratio_ = np.array([np.abs(self.explained_variance_[i] ** 2) / sum_ for i in range(len(self.explained_variance_))])
        sorted_eig_ind = np.argsort(-1 * self.explained_variance_ratio_)
        self.eigenvectors_ = self.eigenvectors_[:, sorted_eig_ind]
        self.explained_variance_ = self.explained_variance_[sorted_eig_ind]

    def transform(self, X):
        """
        param X: numpy array of shape (M,N)
        return X_proj: numpy array of shape (M,K)
        """
        """TODO use self.explained_variance_ratio_ and self.eigenvectors_
        to project X from dimension N to K"""
        w = self.eigenvectors_[:, :self.K]
        return X.dot(w)
