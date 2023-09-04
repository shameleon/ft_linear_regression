import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_dataset(df: pd.DataFrame):
    """ 2D plot of a dataframe"""
    plt.scatter('km','price' , c = 'blue', alpha = 0.5, data = df)
    plt.title('Car price according to mileage')
    plt.xlabel('mileage (km)')
    plt.ylabel('price ($)')
    plt.axis([0, 2.5e5, 0, 1e4])
    plt.grid(alpha = 0.5)
    plt.show()

def plot_cost_function(x_train:np.ndarray, y_train:np.ndarray):
    """ 3D plot of cost function J(θ0, θ1). It depends on bias and weigth ()
        parameters : works better with normalized data """
    ncols = 100
    biases = np.linspace(-2, 4, ncols)
    weights = np.linspace(-6, 4 , ncols)
    cost = np.zeros(ncols * ncols).reshape(ncols, ncols)
    for i in range(ncols):
        for j in range(ncols):
            predicted_y = biases[i] + weights[j] * x_train
            dy = predicted_y - y_train
            cost[j][i] =  np.mean((dy) ** 2)
    biases, weights = np.meshgrid(biases, weights)
    log_cost = np.log10(cost)
    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    surf = ax.plot_surface(biases, weights, log_cost, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax.set_xlabel('bias (θ0)')
    ax.set_ylabel('weight (θ1)')
    ax.set_zlabel('cost function J(θ0, θ1) (log scaled)')
    fig.colorbar(surf, ax = ax, shrink=0.3, aspect=20)
    plt.show()

def plot_gradient_descent(x_train:np.ndarray, y_train:np.ndarray, y_pred:np.ndarray, \
                            loss:np.ndarray, biases:np.ndarray, weights:np.ndarray):
    fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(24, 6))
    i = 0
    ax[i].plot(x_train, y_train, '+', label='Actual values')
    ax[i].plot(x_train, y_pred, label='Predicted values')
    ax[i].set_title('dataset')
    ax[i].set_xlabel('normalized input')
    ax[i].set_ylabel('normalized output')
    ax[i].legend()
    ax[i].grid(alpha = 0.5)
    #ax[i].text(0.9, 0.15, equation, color="orange", fontsize = 12, ha = 'right', va = 'bottom', alpha = 0.7)
    i +=1
    ax[i].plot(loss)
    ax[i].set_title('Loss')
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel('cost function J(θ0, θ1)')
    i +=1
    ax[i].plot(biases)
    ax[i].set_title('Bias')
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel('θ0')
    i +=1
    ax[i].plot(weights)
    ax[i].set_title('Weight')
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel('θ1')
    plt.show()