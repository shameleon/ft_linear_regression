import numpy as np

"""statistics_utils.py:

Usage:
import statistics_utlis as stats

- provides metrics for linear regression analysis
  and a model accuracy report
"""

__author__ = "jmouaike"


""" ################ statistics utils ################ """


class StatisticLinearRegression:
    """Linear regression hypothesis : price = θ0 + θ1 * mileage
        theta[θ0 , θ1] is here calculated with statistical tools
    """
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray):
        self.x = x_train
        self.y = y_train

    def calculate_params(self) -> None:
        theta = np.zeros(2)
        x = self.x
        y = self.y
        a = np.sum(np.multiply(x, y))
        b = np.sum(y) * np.sum(x) / len(x)
        c = np.sum(np.multiply(x, x))
        d = np.sum(x) * np.sum(x) / len(x)
        theta[1] = (a - b) / (c - d)
        theta[0] = np.mean(y) - theta[1] * np.mean(x)
        print('\nStatistical model for linear regression:')
        print('Parameters were directly calculated from the data:')
        print('intercept = {:.1f} \t slope = {:5f}'.format(theta[0], theta[1]))
        y_pred = theta[0] + x * theta[1]
        model_accuracy(y, y_pred)


""" ################ Normalization and reverse way ################ """


def normalize(arr: np.ndarray) -> np.ndarray:
    """ Normalization rescales the values into a range of [0,1].
    Also called min-max scaled """
    span = np.max(arr) - np.min(arr)
    return (arr - np.min(arr)) / span


def normalize_element(norm_element, arr: np.ndarray):
    span = np.max(arr) - np.min(arr)
    return norm_element * span + np.min(arr)


def denormalize_array(normed_arr: np.ndarray,
                      original_arr: np.ndarray) -> np.ndarray:
    span = np.max(original_arr) - np.min(original_arr)
    return normed_arr * span + np.min(original_arr)


def denormalize_element(norm_element, original_arr: np.ndarray):
    span = np.max(original_arr) - np.min(original_arr)
    return norm_element * span + np.min(original_arr)


""" ################ Model Metrics ################ """


def correlation_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    """ Pearson Product-Moment Correlation Coefficient.
    Measure the strength of the linear relationship between
    two quantitative variables.
    Parameters : input and output are numpy np.ndarray
    1D-array of same shape (n, ).
    Return : float whose value is within [-1, 1] interval
    """
    rho = np.corrcoef(x, y)[0, 1]
    print("Pearson correlation coefficient : {:.4f}".format(rho))
    result = ['very weak', 'weak', 'moderate', 'strong']
    strength = (abs(rho) >= 0.3) * 1
    strength += (abs(rho) >= 0.5) * 1
    strength += (abs(rho) >= 0.7) * 1
    print("Dataset has a {} linear correlation".format(result[strength]))
    return rho


def mean_absolute_error(y: np.ndarray, y_pred: np.ndarray) -> float:
    """ MAE """
    absolute_error = abs(y_pred - y)
    mae = np.sum(absolute_error) / len(y)
    return mae


def mean_absolute_percentage_error(y: np.ndarray, y_pred: np.ndarray) -> float:
    """ MAPE cannot be used if there are zero or close-to-zero values """
    mean_absolute_error = abs((y_pred - y) / y) / len(y)
    mape = np.sum(mean_absolute_error) * 100
    return mape


def residual_of_sum_square(y: np.ndarray, y_pred: np.ndarray) -> float:
    """ RSS """
    residual = y_pred - y
    rss = np.sum(residual ** 2)
    return rss


def mean_squared_error(y: np.ndarray, y_pred: np.ndarray) -> float:
    """ cost = MSE / 2 """
    residual = y_pred - y
    rss = np.sum(residual ** 2)
    mse = rss / len(y)
    return mse


def root_mean_squared_error(y: np.ndarray, y_pred: np.ndarray) -> float:
    """ RMSE = sqrt(MSE)"""
    residual = y_pred - y
    rss = np.sum(residual ** 2)
    rmse = np.sqrt(rss / len(y))
    return rmse


def mean_error(y: np.ndarray, y_pred: np.ndarray) -> float:
    error = y_pred - y
    me = np.sum(error) / len(y)
    return me


def r2score(y: np.ndarray, y_pred: np.ndarray) -> float:
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


""" ################ Model Metrics accuracy full report ################ """


def model_accuracy(y_output: np.ndarray, y_pred: np.ndarray):
    """ prints ou a model accuracy report """
    funcs = {'MAE': mean_absolute_error,
             'MAPE': mean_absolute_percentage_error,
             'MSE': mean_squared_error,
             'RMSE': root_mean_squared_error,
             'ME': mean_error,
             'R2 score': r2score}
    color = {'ME': '\x1b[38:5:208m', 'R2 score': '\x1b[38:5:78m'}
    for key in funcs:
        col = ''
        if key in color:
            col = color[key]
        print(f'{col}{key}' + ' = {:.4f}'.format(funcs[key](y_output, y_pred)))
    print('\x1b[0m' + 20 * '_')
