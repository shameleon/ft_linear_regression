import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

def overplot(func):
    """ """
    pass

# @overplot
def plot_dataset(x_input:np.ndarray, y_output:np.ndarray, y_pred:np.ndarray):
    """ 2D plot of a dataframe"""
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax[0].plot(x_input, y_output, '+', c = 'blue', alpha = 0.5, label='Actual values')
    ax[0].plot(x_input, y_pred, c = 'orange', alpha = 0.7, label='Predicted values')
    ax[0].set_title('Car price according to mileage')
    ax[0].set_xlabel('mileage (km)')
    ax[0].set_ylabel('price ($)')
    ax[0].axis([0, 2.5e5, 0, 1e4])
    ax[0].grid(alpha = 0.5)
    y_residual = y_pred -y_output
    # ax[1].xcorr(x_input, y_residual, usevlines=True, maxlags=50, normed=True, lw=2)
    ax[1].stem(x_input, - y_residual)
    plt.show()

def estimate_price(x_input:np.ndarray) -> np.ndarray:
    model_file = "./gradient_descent_model/theta.csv"
    try:
        theta = np.loadtxt(model_file)
    except:
        print('Error: Linear regression model parameters not found.')
    return theta[0] + x_input * theta[1]


if __name__ == "__main__":
    try:
        df = pd.read_csv(f'data.csv')
    except:
        print('Error: could not open file. No data, no model to train.')
        exit(0)
    df = df.dropna()
    arr = df.to_numpy()
    x_input = arr[:,0]
    y_output = arr[:,1]
    y_pred = estimate_price(x_input)
    plot_dataset(x_input, y_output, y_pred)