import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.datasets import make_gaussian_quantiles
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


class PerceptronClassifier:
    def __init__(self, dims=2, epochs=10, eta=0.005):
        self.dims = dims
        self.epochs = epochs
        self.eta = eta
        self.w = np.random.rand(
            (self.dims + 1),
        ).reshape(1, self.dims + 1)

    @np.vectorize
    def _prob_to_label(y):

        if y >= 0:
            return 1
        return 0

    def _derivative(self, y, x):

        if y >= 0:
            return x
        return -x

    def fit(self, x, y=None):
        _x = np.array(x, copy=True)
        _y = np.array(y, copy=True)
        _x, _y = shuffle(_x, _y, random_state=0)

        # add bias dimension
        _x = np.insert(_x, 0, 1, axis=1)

        # training loop
        for epoch in range(self.epochs):
            error = 0

            for x_point, y_point in zip(_x, _y):

                # forward pass
                y_hat = np.matmul(x_point, self.w.T)

                # backward pass
                x_update = np.array(
                    [self._derivative(yh, xb) for yh, xb in zip(y_hat, x_point)]
                )
                self.w = self.w - self.eta * x_update

                # batch error
                error = error + np.sum(np.abs(y_point - y_hat))

            print(
                "epoch: {} - error: {:.3f}, w: {}".format(
                    epoch, (error / _x.shape[0]), self.w
                )
            )
        return self

    def predict(self, x):
        _x = np.array(x, copy=True)
        _x = np.insert(_x, 0, 1, axis=1)
        y_hat = np.dot(_x, self.w.T)
        labels = self._prob_to_label(y_hat)
        return labels

    def score(self, x, y=None):
        y_pred = self.predict(x)
        score = accuracy_score(y, y_pred, normalize=True)
        return score

    def get_params(self, deep=True):
        return {
            "params": self._params,
            "dims": self.dims,
            "epochs": self.epochs,
            "eta": self.eta,
        }

    def set_params(self, dims, epochs, eta):
        self.dims = dims
        self.epochs = epochs
        self.eta = eta
        return self


class PinvRegression:
    def __init__(self):
        self.w = None

    def fit(self, x, y):
        _x = np.array(x, copy=True)
        _x = np.insert(_x, 0, 1, axis=1)
        inverse = np.linalg.inv(np.dot(_x.T, _x))
        self.w = np.dot(np.dot(inverse, _x.T), y)
        return self

    def predict(self, x):
        _x = np.array(x, copy=True)
        _x = np.insert(_x, 0, 1, axis=1)
        y_hat = np.dot(_x, self.w.T)
        return y_hat

    def score(self, x, y=None):
        y_pred = self.predict(x)
        score = mean_squared_error(y, y_pred, normalize=False)
        return score


def plot_decision_boundary(x, y, model_class, **params):

    reduced = x[:, :2]
    model = model_class(**params)
    model.fit(reduced, y)

    h = 0.02  # spatial distance of the mesh

    x_min, x_max = reduced[:, 0].min() - 1, reduced[:, 0].max() + 1
    y_min, y_max = reduced[:, 1].min() - 1, reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    mesh = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh).reshape(xx.shape)

    cmap = matplotlib.colors.ListedColormap(["tab:blue", "tab:orange"])
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap)
    plt.xlabel("x1", fontsize=15)
    plt.ylabel("x2", fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title("Linear classification with Perceptron")
    plt.text(
        x_min + 1,
        y_min + 1,
        "y = sign({:.3f}x1 + {:.3f}x2 + {:.3f})".format(model.w[0][1], model.w[0][2], model.w[0][0]),
        fontsize=14,
    )
    plt.show()


def plot_regression(x, y, model_class, **params):

    model = model_class(**params)
    model.fit(x, y)
    print("regression w: ", model.w)

    y_hat = model.predict(x)

    plt.plot(x[:, 0], y_hat, color="blue")
    plt.scatter(x[:, 0], y, color="tab:orange")
    plt.xlabel("x", fontsize=15)
    plt.ylabel("y", fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title("Linear regression with Pseudo-Inverse")
    ax = plt.axes()
    ax.set_facecolor("#a5c8e1")
    plt.text(
        x.min() + 1,
        y.min() + 1,
        "y = {:.3f}x + {:.3f}".format(model.w[1], model.w[0]),
        fontsize=14,
    )
    plt.show()


def main():

    print("\n *** generating data *** \n")

    # 1. data generation

    x1, y1 = make_classification(
        n_samples=10000,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=3,
        flip_y=0,
        weights=[0.5, 0.5],
        random_state=0,
        shuffle=True,
    )

    x2, y2, coef = make_regression(
        n_samples=100, n_features=1, bias=200, noise=10, coef=True
    )

    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
    sns.scatterplot(x1[:, 0], x1[:, 1], hue=y1, ax=ax1)
    ax1.set_title("Linearly separable data")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")

    sns.scatterplot(x2[:, 0], y2, ax=ax2)
    ax2.set_title("Linear regression data")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    plt.show()

    print("\n *** perceptron classification *** \n")
    plot_decision_boundary(x1, y1, PerceptronClassifier)

    print("\n *** regression with inverse pseudo tranform *** \n")
    plot_regression(x2, y2, PinvRegression)


if __name__ == "__main__":
    main()