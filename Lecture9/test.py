import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Lecture9.lda_skeleton import LDA

X = pd.read_csv('ionosphere.csv', header=None)
target_col = 34
X.loc[X[target_col] == 'b', target_col] = 0
X.loc[X[target_col] == 'g', target_col] = 1
X[target_col] = X[target_col].astype(int)
X.drop(1, axis=1, inplace=True)
for feature in X.columns.difference([target_col]):
    if np.abs(X[[feature, target_col]].corr().iloc[0, 1]) < 0.1:
        X.drop(feature, axis=1, inplace=True)

y = X[target_col]
X.drop(target_col, inplace=True, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
best_C = 1


lda = LDA(K=3)
lda.fit(X_train.values, y_train.values)
X_proj = lda.transform(X_train.values)
print()