from collections import Counter
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

class KNN:
    def __init__(self, k=3, distance='euclidean'):
        self.k = k
        self.distance = distance
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            # Compute distances between the test point and all training points
            if self.distance == 'euclidean':
                distances = np.linalg.norm(self.X_train - x, axis=1)
            elif self.distance == 'manhattan':
                distances = np.sum(np.abs(self.X_train - x), axis=1)
            else:
                distances = np.power(np.sum(np.power(np.abs(self.X_train - x), self.distance), axis=1), 1/self.distance)
                
            # Select the k nearest neighbors
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_indices]
            
            # Assign the class label that appears most frequently among the k nearest neighbors
            label = Counter(nearest_labels).most_common(1)[0][0]
            y_pred.append(label)
        
        return np.array(y_pred)
    

# Load the iris dataset
iris = load_iris()
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
def test():
    # Create a KNN classifier with k=5 and euclidean distance
    knn = KNN(k=5, distance='euclidean')

    # Train the classifier on the training data
    knn.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = knn.predict(X_test)

    # Compute the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Create scatter plots of the test data with colored points representing the true and predicted labels
    fig, ax = plt.subplots()
    scatter1 = ax.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], c='b', cmap='viridis', label=iris.target_names[0])
    scatter2 = ax.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], c='g', cmap='viridis', label=iris.target_names[1])
    scatter3 = ax.scatter(X_test[y_test==2, 0], X_test[y_test==2, 1], c='r', cmap='viridis', label=iris.target_names[2])
    scatter4 = ax.scatter(X_test[:, 0], X_test[:, 1], c='k', cmap='viridis', marker='x', label='Predicted Label')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('KNN Classifier Results')
    handles = [scatter1, scatter2, scatter3, scatter4]
    labels = [h.get_label() for h in handles]
    ax.legend(handles=handles, labels=labels)
    plt.show()

if __name__ == "__main__":
    test()