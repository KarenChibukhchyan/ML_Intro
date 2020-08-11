import numpy as np

class LDA(object):

    def __init__(self, K):
        self.K = K
        self.explained_variance_ = np.array([])
        self.explained_variance_ratio_ = np.array([])
        self.eigenvectors = np.array([])
    
    def compute_scatters(self, X, y):
        S_w = np.zeros((X.shape[1],X.shape[1]))
        S_b = np.zeros((X.shape[1],X.shape[1]))
        mu = X.mean(axis=0)
        for k in np.unique(y):
            class_ind = y==k
            X_k = X[class_ind]
            mu_k = X_k.mean(axis=0)
            X_k_0_mean = X_k - mu_k
            mu_diff = (mu_k - mu).reshape(-1,1)
            S_w += X_k_0_mean.T.dot(X_k_0_mean)
            S_b += class_ind.sum() * mu_diff.dot(mu_diff.T)
        return S_w, S_b
    
    def fit(self, X, y):
        S_w, S_b = self.compute_scatters(X, y)
        S_w_inv = np.linalg.inv(S_w)
        eigenvalues, eigenvectors = np.linalg.eig(S_w_inv.dot(S_b))
        self.explained_variance_ = np.abs(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / self.explained_variance_.sum()
        self.eigenvectors = eigenvectors

    def transform(self, X):
        if self.eigenvectors.shape[0] == 0:
            raise TypeError("Please fit before calling transform")
        sorted_eig_ind = np.argsort(-1 * self.explained_variance_ratio_)
        w = self.eigenvectors[:,sorted_eig_ind][:, :self.K]
        return X.dot(w)


