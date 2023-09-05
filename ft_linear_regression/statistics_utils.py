import numpy as np

class StatisticLinearRegression:
    """Linear regression hypothesis : price = θ0 + θ1 * mileage
        theta[θ0 , θ1] is here calculated with statistical tools
    """
    def __init__(self, x_train, y_train):
        self.x = x_train
        self.y = y_train
        self.calculate_params()        

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
        print('\nStatistical model for linear regression.')
        print('Parameters were directly calculated from the data:')
        print('intercept = {:.1f} \t slope = {:5f}'.format(theta[0], theta[1]))
        y_pred = theta[0] + x * theta[1]
        model_accuracy(y, y_pred, theta)


def correlation_coefficient(x: np.ndarray, y:np.ndarray) -> float:
    """ Pearson Product-Moment Correlation Coefficient.
    Measure the strength of the linear relationship between two quantitative variables.
    Parameters : input and output are numpy np.ndarray 1D-array of same shape (n, )
    Return : float whose value is within [-1, 1] interval
    """
    rho = np.corrcoef(x, y)[0, 1]
    print("Pearson correlation coefficient : {:.4f}".format(rho))
    result = ['very weak', 'weak', 'moderate', 'strong'] 
    strength = (abs(rho) >= 0.3) * 1 + (abs(rho) >= 0.5) * 1 + (abs(rho) >= 0.7) * 1
    print("Dataset has a {} linear correlation".format(result[strength]))
    return rho

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

def normalize(arr: np.ndarray) -> np.ndarray:
    """ Normalization rescales the values into a range of [0,1]. Also called min-max scaled """
    span = np.max(arr) - np.min(arr)
    return (arr - np.min(arr)) / span

def denormalize_array(normed_arr: np.ndarray, y) -> np.ndarray:
    span = np.max(y) - np.min(y)
    return (normed_arr * span) + np.min(y)

def denormalize_element(norm_element, y: np.ndarray):
    span = np.max(y) - np.min(y)
    return (norm_element * span) + np.min(y)

def model_accuracy(y_output:np.ndarray, y_pred:np.ndarray, theta):
    mae = mean_absolute_error(y_output, y_pred)
    mape = mean_absolute_percentage_error(y_output, y_pred)
    print('MAE = {:.3f} \t\t MAPE = {:.3f}%'.format(mae, mape))
    mse = mean_squared_error(y_output, y_pred)
    rmse = root_mean_squared_error(y_output, y_pred)
    print('MSE = {:.3f} \t RMSE = {:.3f}'.format(mse, rmse))
    me = mean_error(y_output, y_pred)
    print('\x1b[38:5:208mMean error ME = {:.3f}\x1b[0m'.format(me))
    r2_score = r2score(y_output, y_pred)
    print('\x1b[38:5:78m     R2 score = {:.3f}\x1b[0m'.format(r2_score))