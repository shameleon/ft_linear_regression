import numpy as np
import pandas as pd

class StatisticLinearRegression:
    """ """
    def __init__(self, x_train, y_train):
        self.x = x_train
        self.y = y_train        

    def calculate_theta(self):
        theta = np.zeros(2)
        x = self.x
        y = self.y
        a = np.sum(np.multiply(x, y))
        b = np.sum(y) * np.sum(x) / len(x)
        c = np.sum(np.multiply(x, x))
        d = np.sum(x) * np.sum(x) / len(x)
        weigth = (a - b) / (c - d)
        bias = np.mean(y) - weigth * np.mean(x)
        print('Statistical model for linear regression.')
        print('parameters directly calculated from the data:')
        print('intercept = {:.1f} \t slope = {:5f}'.format(bias, weigth))
        theta[0] = bias
        theta[1] = weigth
        return theta
    
def mean_absolute_error(y, y_pred):
    absolute_error = abs(y - y_pred)
    mae = np.sum(absolute_error) / len(y)
    return mae

def mean_absolute_percentage_error(y, y_pred):
    mean_absolute_error = abs((y - y_pred) / y) / len(y)
    mape =  np.sum(mean_absolute_error) * 100 
    return mape


def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(f'{file}')
    return df

def update_predicted_output(x_train, theta):
    predicted_output = x_train * theta[1] + theta[0]
    return predicted_output

def main():
    df = load_data('data.csv')
    labels = df.keys()
    arr = df.to_numpy()
    x_train = arr[:,0]
    y_train = arr[:,1]
    statistic_model = StatisticLinearRegression(x_train, y_train)
    theta = statistic_model.calculate_theta()
    y_pred = update_predicted_output(x_train, theta)
    mae = mean_absolute_error(y_train, y_pred)
    mape = mean_absolute_percentage_error(y_train, y_pred)
    print('MAE = {:.3f} \t MAPE = {:.3f}%'.format(mae, mape))
    # df_theta = pd.DataFrame(theta)
    # print(df_theta)
    # df_theta.to_csv("./model/theta.csv")
    np.savetxt("./statistical_model/theta.csv", theta, delimiter=",")

if __name__ == "__main__":
    main()
