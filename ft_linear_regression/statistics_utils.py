import numpy as np

def mean_absolute_error(y, y_pred):
    absolute_error = abs(y - y_pred)
    mae = np.sum(absolute_error) / len(y)
    return mae

def mean_absolute_percentage_error(y, y_pred):
    """ MAPE cannot be used if there are zero or close-to-zero values """
    mean_absolute_error = abs((y - y_pred) / y) / len(y)
    mape =  np.sum(mean_absolute_error) * 100 
    return mape

def residual_of_sum_square(y, y_pred):
    residual = y - y_pred
    rss = np.sum(residual ** 2)
    return rss

def mean_squared_error(y, y_pred):
    residual = y - y_pred
    rss = np.sum(residual ** 2)
    mse = rss / len(y)
    return mse

def root_mean_squared_error(y, y_pred):
    residual = y - y_pred
    rss = np.sum(residual ** 2)
    rmse = np.sqrt(rss / len(y))
    return rmse

def normalize(arr: np.ndarray) -> np.ndarray:
    """ Normalization rescales the values into a range of [0,1]. Also called min-max scaled """
    span = np.max(arr) - np.min(arr)
    return (arr - np.min(arr)) / span

def denormalize_array(normed_arr: np.ndarray, y) -> np.ndarray:
    span = np.max(y) - np.min(y)
    return (normed_arr * span) + np.min(y)

def denormalize_element(norm_element, y):
    span = np.max(y) - np.min(y)
    return (norm_element * span) + np.min(y)