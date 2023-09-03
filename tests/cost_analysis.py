"""custom shaded 3d surface"""

from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def cost_function():
    pass

def normalize(arr: np.ndarray) -> np.ndarray:
    """ Normalization rescales the values into a range of [0,1]. Also called min-max scaled """
    span = np.max(arr) - np.min(arr)
    return (arr - np.min(arr)) / span

def main():
    """
    https://matplotlib.org/stable/gallery/mplot3d/custom_shaded_3d_surface.html
    https://matplotlib.org/stable/gallery/mplot3d/surface3d.html
    """
    # Load and format data
    df = pd.read_csv(f'data.csv')
    labels = df.keys()
    arr = df.to_numpy()
    x_input = normalize(arr[:,0])
    y_output = normalize(arr[:,1])
    
    ncols = 100
    # biases = np.linspace(6000, 10000, ncols)
    # weights = np.linspace(-2.1, 2 , ncols)
    #biases = np.linspace(-10, 10, ncols)
    #weights = np.linspace(-10, 10 , ncols)
    biases = np.linspace(-2, 4, ncols)
    weights = np.linspace(-6, 4 , ncols)
    # predicted_y = self.theta[0] + self.x * self.theta[1]
    #        dy = predicted_y - self.y
    #        self.cost =  np.mean((dy) ** 2)
    cost = np.zeros(ncols * ncols).reshape(ncols, ncols)
    for i in range(ncols):
        for j in range(ncols):
            # print(biases[i], weights[j])
            predicted_y = biases[i] + weights[j] * x_input
            dy = predicted_y - y_output
            cost[j][i] =  np.mean((dy) ** 2)
    biases, weights = np.meshgrid(biases, weights)
    log_cost = np.log10(cost)
    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    surf = ax.plot_surface(biases, weights, log_cost, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    #ax.set_zscale("log", base=10)
    ax.set_xlabel('bias (θ0)')
    ax.set_ylabel('weight (θ1)')
    ax.set_zlabel('cost function J(θ0, θ1) (log scaled)')
    #ax.set_zlim(-1.01, 1.01)
    fig.colorbar(surf, ax = ax, shrink=0.3, aspect=20)
    plt.show()
    # rotating a 3D plot
    # https://matplotlib.org/stable/gallery/mplot3d/rotate_axes3d_sgskip.html


if __name__ == "__main__":
    main()