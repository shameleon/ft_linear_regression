import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd


def plot_dataset(df: pd.DataFrame):
    """ 2D plot of a dataframe """
    plt.scatter('km', 'price', c='blue', alpha=0.5, data=df)
    plt.title('Car price according to mileage')
    plt.xlabel('mileage (km)')
    plt.ylabel('price ($)')
    plt.axis([0, 2.5e5, 0, 1e4])
    plt.grid(alpha=0.5)
    plt.show()


def plot_cost_function(x_train: np.ndarray, y_train: np.ndarray):
    """ 3D plot of cost function J(θ0, θ1). It depends on bias and weigth ()
        parameters : works better with normalized data """
    ncols = 100
    biases = np.linspace(-2, 4, ncols)
    weights = np.linspace(-6, 4, ncols)
    cost = np.zeros(ncols * ncols).reshape(ncols, ncols)
    for i in range(ncols):
        for j in range(ncols):
            predicted_y = biases[i] + weights[j] * x_train
            dy = predicted_y - y_train
            cost[j][i] = np.mean((dy) ** 2)
    biases, weights = np.meshgrid(biases, weights)
    log_cost = np.log10(cost)
    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    surf = ax.plot_surface(biases, weights, log_cost, rstride=1,
                           cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_xlabel('bias (θ0)')
    ax.set_ylabel('weight (θ1)')
    ax.set_zlabel('cost function J(θ0, θ1) (log scaled)')
    fig.colorbar(surf, ax=ax, shrink=0.3, aspect=20)
    plt.show()


def plot_gradient_descent(x_train: np.ndarray, y_train: np.ndarray,
                          y_pred: np.ndarray, loss: np.ndarray,
                          biases: np.ndarray, weights: np.ndarray):
    """ plot 1. trained dataset scatter plot,
        with an additional line to the predicted model output
        plot 2. cost function J[θ0, θ1] over epochs
        plots 3 and 4. resp. biais and weight over epochs

        Parameters :
            x_train and y_train are the trained dataset,
            they might be normalized and reversely.
            Thus, the model parameters would be related to normalized data.
    """
    fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(24, 6))
    i = 0
    ax[i].plot(x_train, y_train, '+', label='Actual values')
    ax[i].plot(x_train, y_pred, label='Predicted values')
    ax[i].set_title('Trained dataset')
    ax[i].set_xlabel('input')
    ax[i].set_ylabel('output')
    ax[i].legend()
    ax[i].grid(alpha=0.5)
    i += 1
    ax[i].plot(loss)
    ax[i].set_title('Loss')
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel('cost function J(θ0, θ1)')
    i += 1
    ax[i].plot(biases)
    ax[i].set_title('Bias')
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel('θ0')
    i += 1
    ax[i].plot(weights)
    ax[i].set_title('Weight')
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel('θ1')
    plt.show()


def plot_final(x_train: np.ndarray, y_train: np.ndarray, y_pred: np.ndarray,
               suptitle: str, title: str):
    """ plot 1. trained dataset scatter plot,
        with an additional line to the predicted model output
        plot 2. cost function J[θ0, θ1] over epochs
        plots 3 and 4. resp. biais and weight over epochs
    """
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 20))
    fig.suptitle(suptitle)
    ax[0].plot(x_train, y_train, '+', label='Actual trained values')
    ax[0].plot(x_train, y_pred, label='Predicted values')
    ax[0].set_title(title, fontsize=10)
    ax[0].set_xlabel('mileage (km)')
    ax[0].set_ylabel('price ($)')
    ax[0].grid(alpha=0.5)
    residual = y_train - y_pred
    ax[1].stem(x_train, residual)
    ax[1].set_xlabel('mileage (km)')
    ax[1].set_ylabel('residual = predicted - actual')
    plt.show()
