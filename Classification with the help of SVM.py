"""Project name: description"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def plot_decision_regions(estimator, x_test, y_test, title=None):
    """
    Plot decision domains for a binary classification problem.
    """
    # Create a grid of points for classification
    xx, yy = np.meshgrid(np.linspace(x_test[:, 0].min() - 0.5, x_test[:, 0].max() + 0.5, 100),
                         np.linspace(x_test[:, 1].min() - 0.5, x_test[:, 1].max() + 0.5, 100))
    z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    # Calculate indicators
    y_pred_local = estimator.predict(x_test)
    accuracy_local = accuracy_score(y_test, y_pred_local)
    precision_local = precision_score(y_test, y_pred_local)
    recall_local = recall_score(y_test, y_pred_local)
    f1_local = f1_score(y_test, y_pred_local)

    # Plot Decision Regions and Data Points
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, z, alpha=0.4)
    plt.scatter(x_test[y_test == 0][:, 0], x_test[y_test == 0][:, 1], marker='s', c='b', s=30, label='Class 0')
    plt.scatter(x_test[y_test == 1][:, 0], x_test[y_test == 1][:, 1], marker='^', c='r', s=30, label='Class 1')

    # Add Support Vector Markers
    plt.scatter(estimator.support_vectors_[:, 0], estimator.support_vectors_[:, 1], s=100,
                linewidths=1, facecolors='none', edgecolors='k')

    # Add text for metrics
    metrics_text = f"Accuracy: {accuracy_local:.2f}\nPrecision: " \
                   f"{precision_local:.2f}\nRecall: {recall_local:.2f}\nF1-score: {f1_local:.2f}"
    plt.text(0.95, 0.45, metrics_text,
             verticalalignment='top', horizontalalignment='right',
             transform=plt.gca().transAxes,
             color='black', fontsize=12)

    # Add text for number of iterations
    plt.text(0.95, 0.05, f"Number of iterations: {estimator.n_iter_[0]}",
             verticalalignment='bottom', horizontalalignment='right',
             transform=plt.gca().transAxes,
             color='black', fontsize=12)

    # Add legend
    plt.legend(loc='upper right')

    if title:
        plt.title(title)
    plt.show()


# Generate dataset
x_train, y_train = make_moons(n_samples=100, noise=0.2, random_state=100)

# Linear SVM with C=1.0
linear_model = svm.SVC(kernel='linear', C=1.0)
linear_model.fit(x_train, y_train)
y_pred_linear = linear_model.predict(x_train)
accuracy_linear = accuracy_score(y_train, y_pred_linear)
precision_linear = precision_score(y_train, y_pred_linear)
recall_linear = recall_score(y_train, y_pred_linear)
f1_linear = f1_score(y_train, y_pred_linear)
print('Linear SVM with C=1.0:')
print('Accuracy:', accuracy_linear)
print('Precision:', precision_linear)
print('Recall:', recall_linear)
print('F1-score:', f1_linear)

# Polynomial SVM with degree=4 and C=1.0
poly_model = svm.SVC(kernel='poly', degree=4, C=1.0)
poly_model.fit(x_train, y_train)
y_pred_poly = poly_model.predict(x_train)
accuracy_poly = accuracy_score(y_train, y_pred_poly)
precision_poly = precision_score(y_train, y_pred_poly)
recall_poly = recall_score(y_train, y_pred_poly)
f1_poly = f1_score(y_train, y_pred_poly)
print('Polynomial SVM with degree=4 and C=1.0:')
print('Accuracy:', accuracy_poly)
print('Precision:', precision_poly)
print('Recall:', recall_poly)
print('F1-score:', f1_poly)

# RBF SVM with gamma=2 and C=1.0
rbf_model = svm.SVC(kernel='rbf', gamma=2, C=1.0)
rbf_model.fit(x_train, y_train)
y_pred_rbf = rbf_model.predict(x_train)
accuracy_rbf = accuracy_score(y_train, y_pred_rbf)
precision_rbf = precision_score(y_train, y_pred_rbf)
recall_rbf = recall_score(y_train, y_pred_rbf)
f1_rbf = f1_score(y_train, y_pred_rbf)
print('RBF SVM with gamma=2 and C=1.0:')
print('Accuracy:', accuracy_rbf)
print('Precision:', precision_rbf)
print('Recall:', recall_rbf)
print('F1-score:', f1_rbf)

# Plot decision regions
plot_decision_regions(linear_model, x_train, y_train, title='Linear SVM with C=1.0')
plot_decision_regions(poly_model, x_train, y_train, title='Polynomial SVM with degree=4 and C=1.0')
plot_decision_regions(rbf_model, x_train, y_train, title='RBF SVM with gamma=2 and C=1.0')
