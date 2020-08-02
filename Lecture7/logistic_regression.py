import numpy as np
import pandas as pd


class SimpleLogisticRegression:

    def __init__(self, threshold=0.5, lambda_=0, epsilon=1e-6, learning_rate=1e-4, max_steps=500_000):
        self.lambda_ = lambda_
        self.threshold = threshold
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.max_steps = max_steps

    def fit(self, X, Y):
        self.gradient_descent(X, Y)

    def predict(self, X):
        sigmoid = self.sigmoid(X, self.beta)
        return pd.Series(np.where(sigmoid > self.threshold, 1, 0))

    def get_beta(self):
        return self.beta

    def sigmoid(self, X, beta):
        dot = np.dot(X, beta)
        p = 1 / (1 + np.exp(-dot))
        return p

    def logistic_function(self, X, beta):
        sigmoid = self.sigmoid(X, beta)
        return np.where(sigmoid > self.threshold, 1, 0)

    def gradient(self, beta, X, Y):
        diff = self.sigmoid(X, beta) - Y
        gradient = np.dot(X.T, diff)
        return gradient

    def gradient_descent(self, X, Y):
        num_rows, num_columns = X.shape
        beta = np.ones(num_columns + 1)
        X = np.insert(X, 0, 1, axis=1)
        cost_ = self.cost_func(X, Y, beta)
        step = 1
        while step < self.max_steps:
            old_cost_ = cost_
            gradient = self.gradient(beta, X, Y)
            beta[0] = beta[0] - self.learning_rate * (1 / num_rows * gradient[0])

            regularization = (self.lambda_ / num_columns * beta)[1:]
            beta[1:] = beta[1:] - self.learning_rate * (1 / num_rows * gradient[1:] + regularization)

            cost_ = self.cost_func(X, Y, beta)
            if not step%1000:
                print(f'step: {step}    error: {np.abs(cost_ - old_cost_)}')
            if np.abs(cost_ - old_cost_) <= self.epsilon:
                print(f'Gradient Descent converged at {step}-th step')
                break
            step += 1
        else:
            print('Can not converge gradient descent with given epsilon, step size and maximum steps')
            print('Last value of beta was ', str(beta))
        self.beta = beta

    def cost_func(self, X, Y, beta):
        n = len(X)
        d = len(beta) - 1
        sigmoid = self.sigmoid(X, beta)
        regularization = self.lambda_ / 2 * d * (np.sum(beta**2))
        cost = - 1 / n * (np.dot(Y, np.log(sigmoid)) + np.dot((1 - Y), np.log(1 - sigmoid))) + regularization
        return cost
