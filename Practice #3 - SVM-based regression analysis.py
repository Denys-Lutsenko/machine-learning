""" Practice #3 - SVM-based regression analysis"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
matplotlib.use('TkAgg')

np.random.seed(100)
x_test = 2 * np.random.rand(100, 1) - 1
y_test = 0.2 + 0.1 * x_test[:, 0] + 0.5 * x_test[:, 0] ** 2 + np.random.randn(100) / 10

# Linear kernel
svm_reg = make_pipeline(StandardScaler(),
                        SVR(kernel='linear', epsilon=0.01, C=10))
svm_reg.fit(x_test, y_test)
pred = svm_reg.predict(x_test)
print("Linear SVR Train RMSE: %.2f" % np.sqrt(mean_squared_error(y_test, pred)))
print("Linear SVR Train R^2 Score: %.2f" % r2_score(y_test, pred))

# Polynomial kernel
svm_poly_reg = make_pipeline(StandardScaler(),
                             SVR(kernel="poly", degree=3, C=1000, epsilon=0.1))

svm_poly_reg.fit(x_test, y_test)
pred_poly = svm_poly_reg.predict(x_test)
print("Polynomial SVR Train RMSE: %.2f" % np.sqrt(mean_squared_error(y_test, pred_poly)))
print("Polynomial SVR Train R^2 Score: %.2f" % r2_score(y_test, pred_poly))

# RBF kernel
svm_rbf_reg = make_pipeline(StandardScaler(),
                            SVR(kernel="rbf", gamma=0.1, C=1000, epsilon=0.05))

svm_rbf_reg.fit(x_test, y_test)
pred_rbf = svm_rbf_reg.predict(x_test)
print("RBF SVR Train RMSE: %.2f" % np.sqrt(mean_squared_error(y_test, pred_rbf)))
print("RBF SVR Train R^2 Score: %.2f" % r2_score(y_test, pred_rbf))

# Sigmoid kernel
svm_sig_reg = make_pipeline(StandardScaler(),
                            SVR(kernel="sigmoid", gamma=0.1, C=1, epsilon=0.01))
svm_sig_reg.fit(x_test, y_test)
pred_sig = svm_sig_reg.predict(x_test)
print("Sigmoid SVR Train RMSE: %.2f" % np.sqrt(mean_squared_error(y_test, pred_sig)))
print("Sigmoid SVR Train R^2 Score: %.2f" % r2_score(y_test, pred_sig))


def find_support_vectors(svm_model, x_data, y_data):
    """
    Finds the support vectors for a given SVM model.
    """
    y_pred = svm_model.predict(x_data)
    epsilon = svm_model.epsilon
    off_margin = np.abs(y_data - y_pred) >= epsilon
    return np.argwhere(off_margin)


def plot_svm_regression(svm_model, x_data, y_data, ax_limits):
    """
    This function plots the support vector regression results 
    of an SVM model along with the predicted values and the margin of error.
    """
    x_vals = np.linspace(ax_limits[0], ax_limits[1], 100).reshape(100, 1)
    y_pred = svm_model.predict(x_vals)
    svm_est = svm_model.named_steps['svr']
    epsilon = svm_est.epsilon
    plt.plot(x_vals, y_pred, "k-", linewidth=2, label=r"$\hat{y}$", zorder=-2)
    plt.plot(x_vals, y_pred + epsilon, "k--", zorder=-2)
    plt.plot(x_vals, y_pred - epsilon, "k--", zorder=-2)
    support_vectors = find_support_vectors(svm_est, x_data, y_data)
    plt.scatter(x_data[support_vectors], y_data[support_vectors])


# Plotting the results
plt.figure(figsize=(12, 10))
plt.subplot(221)
plot_svm_regression(svm_reg, x_test, y_test, [-1, 1, 0, 1])
plt.title('Linear Kernel')
plt.subplot(222)
plot_svm_regression(svm_poly_reg, x_test, y_test, [-1, 1, 0, 1])
plt.title('Polynomial Kernel')
plt.subplot(223)
plot_svm_regression(svm_rbf_reg, x_test, y_test, [-1, 1, 0, 1])
plt.title('RBF Kernel')
plt.subplot(224)
plot_svm_regression(svm_sig_reg, x_test, y_test, [-1, 1, 0, 1])
plt.title('Sigmoid Kernel')
plt.tight_layout()
plt.show()
