import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression


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
        # initalize weights from N~(0,0.01)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            # ???
            self.w_[0] += self.eta * errors.sum()

            # ??? compute logistic cost now instead of SSE cost
            cost = -y.dot(np.log(output)) - ((1-y).dot(np.log(1-output)))
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calc net input"""
        # ??? why add 0th term to all others? is it adding bias term to all others?
        return np.dot(X, self.w_[1:] + self.w_[0])

    def activation(self, z):
        """Compute logistic sigmoid activation"""
        # clip => if value of z_i goes below -250 it becomes -250
        #                         geos above 250 it becomes 250
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """Return class label after unit step"""
        print(self.net_input(X))
        print(self.net_input(X).shape)
        return np.where(self.net_input(X) >= 0.0, 1, 0)


bc = load_breast_cancer()
bc_df = pd.DataFrame(data=bc.data, columns=bc.feature_names)
bc_df["target"] = bc.target


print("memory size before ", bc_df.memory_usage(deep=True).sum())
bc_df.iloc[:, :-1] = StandardScaler().fit_transform(bc_df.iloc[:, :-1])
bc_df.iloc[:, :-1] = bc_df.iloc[:, :-1].astype(np.float16)
bc_df.iloc[:, -1] = bc_df.iloc[:, -1].astype(np.int8)
print("memory size after ", bc_df.memory_usage(deep=True).sum())

X = bc_df.drop(columns=["target"]).values
y = bc_df["target"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

lr_clf = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
lr_clf.fit(X_train, y_train)
y_pred = lr_clf.predict(X_test)
print(len(y_pred))
print(y_pred)

test_acc = np.sum(np.where(y_test == y_pred, 1, 0)) / len(y_test)
print(f"test accuracy = {round(test_acc, 2)}")