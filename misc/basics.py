import numpy as np

import matplotlib.pyplot as plt 
# Linear regression

# Logistic regression
class LogisticRegression:
    def __init__(self, n_iters, learning_rate):
        self.W = None
        self.n_iters = n_iters
        self.learning_rate = learning_rate

    def _sigmoid(self, z):
        return 1.0 / (1 + np.exp(-z))

    # X is n*m, m features, n points
    def fit(self, X, y):
        n_samples = len(X)
        ones = np.ones((n_samples, 1))
        X = np.hstack([ones, X])
        self.W = np.zeros(X.shape[1])

        for i in range(self.n_iters):
            d = X.T @ (y - self._sigmoid(X @ self.W))
            # self.W += (1.0/n_samples) * self.learning_rate * np.sum(d) # so dumb
            self.W += (1.0/n_samples) * self.learning_rate * d
    
    def predict(self, x):
        ones = np.ones((len(x), 1))
        x = np.hstack([ones, x])
        z = np.dot(x, self.W)
        y_pred = self._sigmoid(z)
        y_pred = np.round(y_pred).astype(int)
        return y_pred

class LogisticRegression1:
    
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        # initialize weights and bias to zeros
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # gradient descent optimization
        for i in range(self.n_iters):
            # calculate predicted probabilities and cost
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(z)
            cost = (-1 / n_samples) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            
            # calculate gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
    def predict(self, X):
        # calculate predicted probabilities
        z = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(z)
        # convert probabilities to binary predictions
        return np.round(y_pred).astype(int)
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
# K-means clustering

# K-nearest neighbors

# Decision trees

# Linear SVM

# Neural networks
'''
Perceptron (code)
FeedForward NN (code)
Softmax (code)
Convolution (code)
'''

# stratified sampling (link)


def test(cls):
    # create sample dataset
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([0, 0, 1, 1, 1])

    # initialize logistic regression model
    lr = cls(n_iters=1000, learning_rate=0.01)

    # train model on sample dataset
    lr.fit(X, y)

    # make predictions on new data
    X_new = np.array([[6, 7], [7, 8]])
    y_pred = lr.predict(X_new)

    print(y_pred)  # [1, 1]

def testplot(cls):
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([0, 0, 1, 1, 1])

    # initialize logistic regression model
    lr = cls(learning_rate=0.01, n_iters=1000)

    # train model on dataset
    lr.fit(X, y)

    # plot decision boundary
    x1 = np.linspace(0, 6, 100)
    x2 = np.linspace(0, 8, 100)
    xx, yy = np.meshgrid(x1, x2)
    Z = lr.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)

    # plot data points
    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Spectral)

    plt.show()

if __name__ == "__main__":
    testplot(LogisticRegression)
    #testplot(LogisticRegression1)
