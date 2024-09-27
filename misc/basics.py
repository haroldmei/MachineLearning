import numpy as np

import matplotlib.pyplot as plt 
# Linear regression

# Logistic regression
class LogisticRegression:
    def __init__(self, epochs, lr):
        self.W = None
        self.epochs = epochs
        self.lr = lr

    def _sigmoid(self, z):
        return 1.0 / (1 + np.exp(-z))

    # X is n*m, m features, n points
    def fit(self, X, y):
        n_samples = len(X)
        ones = np.ones((n_samples, 1))
        X = np.hstack([ones, X])
        self.W = np.zeros(X.shape[1])

        for i in range(self.epochs):
            d = X.T @ (y - self._sigmoid(X @ self.W))
            self.W += (1.0/n_samples) * self.lr * np.sum(d)
    
    def predict(self, x):
        ones = np.ones((len(x), 1))
        x = np.hstack([ones, x])
        return self._sigmoid(x @ self.W)

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


def test():
    # create sample dataset
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([0, 0, 1, 1, 1])

    # initialize logistic regression model
    lr = LogisticRegression(epochs=1000, lr=0.01)

    # train model on sample dataset
    lr.fit(X, y)

    # make predictions on new data
    X_new = np.array([[6, 7], [7, 8]])
    y_pred = lr.predict(X_new)

    print(y_pred)  # [1, 1]


def plot(X, y):
    # Plot the data and the linear regression line
    plt.scatter(X, y, color='blue')
    plt.plot(X, y_pred, color='red')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.show()

if __name__ == "__main__":
    test()
