from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=200, noise=0.3, random_state=200)


model = DecisionTreeClassifier(random_state=50)
model.fit(X, y)

model_restricted = DecisionTreeClassifier(random_state=50, min_samples_leaf=4)
model_restricted.fit(X, y)

def plot_decision_boundary(clf, X, y, axes, cmap):
   x1, x2 = np.meshgrid(np.linspace(axes[0], axes[1], 100),
                        np.linspace(axes[2], axes[3], 100))
   X_new = np.c_[x1.ravel(), x2.ravel()]
   y_pred = clf.predict(X_new).reshape(x1.shape)

   plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=cmap)
   plt.contour(x1, x2, y_pred, cmap="Greys", alpha=0.8)
   colors = {"Wistia": ["#78785c", "#c47b27"], "Pastel1": ["red", "blue"]}
   markers = ("o", "^")
   for idx in (0, 1):
       plt.plot(X[:, 0][y == idx], X[:, 1][y == idx],
                color=colors[cmap][idx], marker=markers[idx], linestyle="none")
   plt.axis(axes)
   plt.xlabel(r"$x_1$")
   plt.ylabel(r"$x_2$", rotation=0)


fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
plt.sca(axes[0])
plot_decision_boundary(model, X, y,
                      axes=[-1.5, 2.4, -1, 1.5], cmap="Wistia")
plt.title("No restrictions")
plt.sca(axes[1])
plot_decision_boundary(model_restricted, X, y,
                      axes=[-1.5, 2.4, -1, 1.5], cmap="Wistia")
plt.title(f"min_samples_leaf = {model_restricted.min_samples_leaf}")
plt.ylabel("")
plt.show()
