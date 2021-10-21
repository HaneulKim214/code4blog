import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LogisticRegressionGD(object):
    """ Logistic regression using gradient descent

    Parameters
    -----------
    eta : float
        Learning rate [0, 1]
    n_iter : int
    random_state : int
        Random number generator seed for random weight
        initialization.


    Attributes
    ----------
    w_ : 1-d array
      Weights after fit.
    cost_ : list
      Logistic cost function value in each epoch
    """
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        Paramters
        ---------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors.
        y : {array-like}, shape = [n_examples]
          Target values.

        Returns
        -------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()

            # ??? compute logistic cost now instead of SSE cost
            cost = -y.dot(np.log(output)) - ((1-y).dot(np.log(1-output)))
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calc net input"""
        return np.dot(X, self.w_[1:] + self.w_[0])

    def activation(self, z):
        """Compute logistic sigmoid activation"""
        # ??? wtf are -250, 250
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)


iris = load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

lrgd = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
lrgd.fit(X_train_01_subset,
         y_train_01_subset)

pred_y = lrgd.predict(X_test_std)

print(X_test_std)
print(pred_y)
print(y_test)

print(np.sum(np.where(pred_y == y_test, 1, 0)) / len(pred_y))