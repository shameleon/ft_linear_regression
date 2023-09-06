import numpy as np
import pandas as pd

def formula_decorator(func):
    def wrapper(arr1:np.ndarray, arr2:np.ndarray):
        residual = arr2 - arr1
        name, res = func(residual)
        print(name, ' = {:.4f}'.format(res))
    return wrapper
    
@formula_decorator
def mean_absolute_error(diff:np.ndarray) -> float:
    return 'MAE', np.sum(abs(diff)) / len(diff)

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
    mean_absolute_error(y_output, y_pred)