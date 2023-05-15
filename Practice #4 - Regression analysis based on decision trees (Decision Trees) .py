"""Practice #4 - Regression analysis based on decision trees (Decision Trees)"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor


def plot_regression_predictions(regressor, input_data, labels, axes_values=None, label=None):
    """Function to plot regression predictions."""
    if axes_values is None:
        axes_values = [-0.5, 0.5, -0.5, 0.5]
    x1 = np.linspace(axes_values[0], axes_values[1], 500).reshape(-1, 1)
    y_pred = regressor.predict(x1)
    plt.axis(axes_values)
    plt.xlabel("$x_1$")
    plt.ylabel("$y$")
    plt.plot(input_data, labels, "b.")
    plt.plot(x1, y_pred, "r.-", linewidth=2, label=label)
    plt.legend()


def main():
    # Generate the dataset
    np.random.seed(100)
    x_quad = np.random.rand(250, 1) - 0.5  # a single random input feature
    y_quad = x_quad ** 3 - 2 * x_quad ** 2 + 0.15 * np.random.randn(250, 1)

    # Use the default settings for the decision tree
    tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_reg1.fit(x_quad, y_quad)
    pred = tree_reg1.predict(x_quad)
    print("DT Train RMSE: %.2f" % np.sqrt(mean_squared_error(y_quad, pred)))
    print("DT Train R^2 Score: %.2f" % r2_score(y_quad, pred))

    # Show the resulting scatterplot
    fig, axes = plt.subplots(ncols=2, figsize=(10, 4))

    plt.sca(axes[0])
    plot_regression_predictions(tree_reg1, x_quad, y_quad, label="max_depth=2")

    # Use different decision trees and show the resulting graphs
    tree_reg2 = DecisionTreeRegressor(max_depth=3, random_state=42)
    tree_reg2.fit(x_quad, y_quad)
    pred = tree_reg2.predict(x_quad)
    print("DT Train RMSE: %.2f" % np.sqrt(mean_squared_error(y_quad, pred)))
    print("DT Train R^2 Score: %.2f" % r2_score(y_quad, pred))

    plt.sca(axes[1])
    plot_regression_predictions(tree_reg2, x_quad, y_quad, label="max_depth=3")

    plt.show()

    # Generate a dataset for validation and calculate metrics to prevent overtraining
    np.random.seed(200)
    x_quad_eval = np.random.rand(250, 1) - 0.5  # a single random input feature
    y_quad_eval = x_quad_eval ** 3 - 2 * x_quad_eval ** 2 + 0.15 * np.random.randn(250, 1)

    pred = tree_reg2.predict(x_quad_eval)
    print("DT Validation RMSE: %.2f" % np.sqrt(mean_squared_error(y_quad_eval, pred)))
    print("DT Validation R^2 Score: %.2f" % r2_score(y_quad_eval, pred))


if __name__ == "__main__":
    main()
