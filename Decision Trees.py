"""Practice #2 - Classification using Decision Trees"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier

# Generate data
features, labels = make_moons(n_samples=200, noise=0.3, random_state=200)

# Train a decision tree without restrictions
model = DecisionTreeClassifier(random_state=50)
model.fit(features, labels)

# Train a decision tree with more fine-grained settings
model_restricted = DecisionTreeClassifier(min_samples_leaf=5, min_samples_split=5, max_depth=4, random_state=50)
model_restricted.fit(features, labels)

# Evaluate the models
train_acc = model.score(features, labels)
test_acc = model.score(features, labels)
print("Model without restrictions:")
print(f"- Training Accuracy: {train_acc:.1%}")
print(f"- Testing Accuracy: {test_acc:.1%}")

train_acc_restricted = model_restricted.score(features, labels)
test_acc_restricted = model_restricted.score(features, labels)
print("Model with restrictions:")
print(f"- Training Accuracy: {train_acc_restricted:.1%}")
print(f"- Testing Accuracy: {test_acc_restricted:.1%}")
print(f"- Maximum Depth: {model_restricted.max_depth}")
print(f"- Minimum Samples to Split: {model_restricted.min_samples_split}")
print(f"- Minimum Samples to be Leaf: {model_restricted.min_samples_leaf}")


def create_new_features(axes_range):
    """ Create new features by applying a grid to the original features """
    x1s = np.linspace(axes_range[0], axes_range[1], 100)
    x2s = np.linspace(axes_range[2], axes_range[3], 100)
    x1_grid, x2_grid = np.meshgrid(x1s, x2s)
    new_features = np.c_[x1_grid.ravel(), x2_grid.ravel()]
    return new_features, x1_grid, x2_grid


def plot_decision_boundary(clf, features_inner, labels_inner, axes_range, cmap):
    """ Plot the decision boundary of a classifier by using contour plots """
    new_features, x1_grid, x2_grid = create_new_features(axes_range)
    y_pred = clf.predict(new_features).reshape(x1_grid.shape)
    plt.contourf(x1_grid, x2_grid, y_pred, alpha=0.3, cmap=cmap)
    plt.contour(x1_grid, x2_grid, y_pred, levels=[0.5], cmap="Greys", alpha=0.8)

    # Plot the data points
    colors = {"viridis": ["#440154", "#FDE724"], "plasma": ["#440154", "#FDE724"]}
    markers = ("o", "^")
    for idx in (0, 1):
        plt.scatter(features_inner[:, 0][labels_inner == idx], features_inner[:, 1][labels_inner == idx],
                    color=colors[cmap][idx], marker=markers[idx], alpha=0.8)
    plt.axis(axes_range)
    plt.xlabel(r"$x_1$", fontsize=14)
    plt.ylabel(r"$x_2$", fontsize=14)


fig, axes = plt.subplots(ncols=2, figsize=(12, 4), constrained_layout=True)

plt.sca(axes[0])
plot_decision_boundary(model, features, labels,
                       axes_range=[-1.5, 2.4, -1, 1.5],
                       cmap="viridis")
plt.xlabel("Feature 1", fontsize=14)
plt.ylabel("Feature 2", fontsize=14)
plt.title("Decision Tree without Restrictions")

plt.sca(axes[1])
plot_decision_boundary(model_restricted, features, labels,
                       axes_range=[-1.5, 2.4, -1, 1.5],
                       cmap="plasma")
plt.xlabel("Feature 1", fontsize=14)
plt.ylabel("Feature 2", fontsize=14)
plt.title("Decision Tree with Restrictions")

custom_legend = [Line2D([0], [0], marker="o", color="w", label="Class 0",
                        markerfacecolor="#440154", markersize=10),
                 Line2D([0], [0], marker="^", color="#FDE724", label="Class 1",
                        markerfacecolor="#FDE724", markersize=10)]

# Add the legend to the plot
plt.legend(handles=custom_legend, loc="upper right")

# Show the plot
plt.show()
