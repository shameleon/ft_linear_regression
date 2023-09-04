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
        weigth = (a - b) / (c - d)
        bias = np.mean(y) - weigth * np.mean(x)
        print('\nStatistical model for linear regression.')
        print('parameters directly calculated from the data:')
        print('intercept = {:.1f} \t slope = {:5f}'.format(bias, weigth))
        y_pred = bias + x * weigth
        mae = mean_absolute_error(y, y_pred)
        mape = mean_absolute_percentage_error(y, y_pred)
        print('MAE = {:.3f} \t MAPE = {:.3f}%'.format(mae, mape))

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