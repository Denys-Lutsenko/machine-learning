import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def plot_decision_regions(estimator, X, y, title=None):
    """
    Plot decision domains for a binary classification problem.
    """
    # Create a grid of points for classification
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100),
                         np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 100))
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Calculate indicators
    y_pred = estimator.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    # Plot Decision Regions and Data Points
    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[y==0][:, 0], X[y==0][:, 1], marker='s', c='b', s=30, label='Class 0')
    plt.scatter(X[y==1][:, 0], X[y==1][:, 1], marker='^', c='r', s=30, label='Class 1')

    # Add Support Vector Markers
    plt.scatter(estimator.support_vectors_[:, 0], estimator.support_vectors_[:, 1], s=100,
                linewidths=1, facecolors='none', edgecolors='k')

    # Add text for metrics
    metrics_text = f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1-score: {f1:.2f}"
    plt.text(0.95, 0.45, metrics_text,
             verticalalignment='top', horizontalalignment='right',
             transform=plt.gca().transAxes,
             color='black', fontsize=12)

    # Add text for number of iterations
    plt.text(0.95, 0.05, f"Number of iterations: {estimator.n_iter_}",
             verticalalignment='bottom', horizontalalignment='right',
             transform=plt.gca().transAxes,
             color='black', fontsize=12)

    # Add legend
    plt.legend(loc='upper right')

    if title:
        plt.title(title)
    plt.show()



# Generate dataset
X, y = make_moons(n_samples=100, noise=0.2, random_state=100)

# Linear SVM with C=1.0
linear_model = svm.SVC(kernel="linear", C=1.0)
linear_model.fit(X, y)
y_pred = linear_model.predict(X)
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
print("Linear SVM with C=1.0:")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1-score: ", f1)

# Polynomial SVM with degree=4 and C=1.0
poly_model = svm.SVC(kernel="poly", degree=4, gamma="scale", C=1.0)
poly_model.fit(X, y)
y_pred = poly_model.predict(X)
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
print("\nPolynomial SVM with degree=4 and C=1.0:")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1-score: ", f1)

# RBF SVM with gamma=10 and C=1.0
rbf_model = svm.SVC(kernel="rbf", gamma=10, C=1.0)
rbf_model.fit(X, y)
y_pred = rbf_model.predict(X)
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
print("\nRBF SVM with gamma=10 and C=1.0:")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1-score: ", f1)

plot_decision_regions(linear_model, X, y, title="Linear SVM with C=1.0")
plot_decision_regions(poly_model, X, y, title="Polynomial SVM with degree=4 and C=1.0")
plot_decision_regions(rbf_model, X, y, title="RBF SVM with gamma=10 and C=1.0")

