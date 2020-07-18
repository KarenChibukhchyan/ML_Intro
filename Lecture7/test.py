import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

diabetes_df = pd.read_csv('diabetes_data.csv')


def drop_cols_with_zeros(df, threshold=0.9):
    new_df = df.replace({0: np.nan})
    new_df.dropna(thresh=0.9 * len(diabetes_df), axis='columns', inplace=True)
    return new_df


def replace_zeros_with_means(df):
    for feature in df.columns:
        nonzero_mean = df.loc[df[feature] != 0, feature].mean()
        df.loc[df[feature] == 0, feature] = nonzero_mean
        df[feature].fillna(nonzero_mean, inplace=True)
    return df


def find_best_features(X, y, threshold=0.2):
    columns_to_drop = []
    for column in X.columns:
        # print(np.corrcoef(X[column].values, y.values)[0, 1])
        if abs(np.corrcoef(X[column].values, y.values)[0, 1]) < threshold:
            columns_to_drop.append(column)
    return X[X.columns.difference(columns_to_drop)]


target_col = 'Outcome'
y = diabetes_df[target_col]
X_original = diabetes_df[diabetes_df.columns.difference([target_col])]
X = drop_cols_with_zeros(X_original)
X = replace_zeros_with_means(X)
X = find_best_features(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

class SimpleLogisticRegression:

    def fit(self, X, Y, threshold=0.5, learning_rate=1e-4):
        # threshold is value by which outcome probabilities are binarizied
        self.threshold = threshold
        self.gradient_descent(X.values, Y.values, step_size=learning_rate)

    def predict(self, X):
        sigmoid = self.sigmoid(X, self.beta)
        return pd.Series(np.where(sigmoid > self.threshold, 1, 0))

    def get_beta(self):
        return self.beta

    def sigmoid(self, X, beta):
        """
        :param X: data matrix (2 dimensional np.array)
        """
        dot = np.dot(X, beta)
        p = 1 / (1 + np.exp(-dot))
        return p

    def gradient(self, beta, X, Y):
        """
        :param X: data matrix (2 dimensional np.array)
        :param Y: response variables (1 dimensional np.array)
        :param beta: value of beta (1 dimensional np.array)
        :return: np.array i.e. gradient according to the data
        """

        gradient = self.sigmoid(X, beta) - Y
        dot = np.dot(gradient, X)
        return dot

    def cost_func(self, X, Y, beta):
        """
        :param X: data matrix (2 dimensional np.array)
        :param Y: response variables (1 dimensional np.array)
        :param beta: value of beta (1 dimensional np.array)
        :return: numberic value of the cost function

        """
        n = len(X)
        hypoth = self.sigmoid(X, beta)
        cost = 1 / n * (np.dot(-Y, np.log(hypoth)) - np.dot( (1 - Y),  np.log(1 - hypoth)))
        return cost

    def gradient_descent(self, X, Y, epsilon=1e-6, step_size=1e-4, max_steps=10_000):
        """
        :param X: data matrix (2 dimensional np.array)
        :param Y: response variables (1 dimensional np.array)
        :param epsilon: threshold for a change in cost function value
        :param max_steps: maximum number of iterations before algorithm will terminate.
        """
        beta = np.zeros(X.shape[1] + 1)
        # adding columns with ones to X
        X = np.insert(X, 0, 1, axis=1)

        step = 1
        while step < max_steps:
            gradient = self.gradient(beta, X, Y)
            beta -= step_size * gradient
            if np.abs(self.cost_func(X, Y, beta)) < epsilon:
                print('Gradient descent converged on {}-th step'.format(str(step)))
                break
            step += 1
        else:
            print('Can not converge gradient descent with given epsilon, step size and maximum steps')
            print('Last value of beta was ', str(beta))

        self.beta = beta


my_clf = SimpleLogisticRegression()
my_clf.fit(X_train, y_train)
X_array = np.insert(X_test.values, 0, 1, axis=1)
my_y_pred = my_clf.predict(X_array)
print('Accuracy of my classifier: ', accuracy_score(y_test, my_y_pred))

sklearn_clf = LogisticRegression()
sklearn_clf.fit(X_train, y_train)
y_pred = sklearn_clf.predict(X_test)
print('Accuracy of sklearn regressor: ', accuracy_score(y_test, y_pred))

