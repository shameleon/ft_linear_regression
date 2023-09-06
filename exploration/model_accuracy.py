import numpy as np
import pandas as pd


def r2score(y:np.ndarray, y_pred:np.ndarray) -> float:
    """
    Description:
    Calculate the R2score between the predicted output and the output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_pred: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        r2score: has to be a float.
    """
    mse = mean_squared_error(y, y_pred)
    var = np.sum((y - np.mean(y)) ** 2) 
    r2score = 1 - mse / var
    return r2score 

def mean_absolute_error(y:np.ndarray, y_pred:np.ndarray) -> float:
    """ MAE """
    absolute_error = abs( y_pred - y)
    mae = np.sum(absolute_error) / len(y)
    return mae

def mean_absolute_percentage_error(y:np.ndarray, y_pred:np.ndarray) -> float:
    """ MAPE cannot be used if there are zero or close-to-zero values """
    mean_absolute_error = abs((y_pred - y) / y) / len(y)
    mape =  np.sum(mean_absolute_error) * 100 
    return mape

def residual_of_sum_square(y:np.ndarray, y_pred:np.ndarray) -> float:
    """ RSS """
    residual = y_pred - y
    rss = np.sum(residual ** 2)
    return rss

def mean_squared_error(y:np.ndarray, y_pred:np.ndarray) -> float:
    """ cost = MSE / 2 """
    residual = y_pred - y
    rss = np.sum(residual ** 2)
    mse = rss / len(y)
    return mse

def root_mean_squared_error(y:np.ndarray, y_pred:np.ndarray) -> float:
    """ RMSE = sqrt(MSE)"""
    residual = y_pred - y
    rss = np.sum(residual ** 2)
    rmse = np.sqrt(rss / len(y))
    return rmse

def mean_error(y:np.ndarray, y_pred:np.ndarray) -> float:
    error = y_pred - y
    me = np.sum(error) / len(y)
    return me

def estimate_price(x_input:np.ndarray) -> np.ndarray:
    model_file = "./gradient_descent_model/theta.csv"
    try:
        theta = np.loadtxt(model_file)
    except:
        print('Error: Linear regression model parameters not found.')
    return theta[0] + x_input * theta[1]

def model_accuracy(y_output:np.ndarray, y_pred:np.ndarray):
    funcs = {'MAE': mean_absolute_error, 'MAPE': mean_absolute_percentage_error, \
                'MSE': mean_squared_error, 'RMSE': root_mean_squared_error, \
                'ME': mean_error, 'R2 score': r2score}
    for key in funcs:
        print(key)

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
    model_accuracy(y_output, y_pred)