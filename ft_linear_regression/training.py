import numpy as np
import pandas as pd
import statistics_utils as stat
from LinearRegressionClass import LinearRegressionGradientDescent

def predict_output(x , theta):
    return theta[0] + x * theta[1]

def get_theta_from_csv(theta):
        """ """
        return f'\033[1;32;47mModel to dataset : \n \
            y = {theta[0]} + x * {theta[1]}\033[0m'

def main():
    df = pd.read_csv(f'data.csv')
    labels = df.keys()
    arr = df.to_numpy()
    x_input = arr[:,0]
    y_output = arr[:,1]
    x_train = stat.normalize(arr[:,0])
    y_train = stat.normalize(arr[:,1])
    normalized_model = LinearRegressionGradientDescent(x_train, y_train)
    normalized_model.train_gradient_descent()
    print(normalized_model)
    norm_theta = normalized_model.get_theta()
    # theta = model.denormalize_theta(y_output)
    y_pred_norm = normalized_model.predict_output()
    # y_pred_norm = predict_output(x_train, model.get_theta())
    y_pred = stat.denormalize_array(y_pred_norm, y_output)
    theta = np.zeros(2)
    theta[0] = stat.denormalize_element(norm_theta[0], y_output)
    theta[1] = (y_pred[-1] - y_pred[0]) / (x_input[-1] - x_input[0])
    np.savetxt("./gradient_descent_model/theta.csv", theta, delimiter=",")
    print(get_theta_from_csv(theta))
    # model accuracy
    mae = stat.mean_absolute_error(y_output, y_pred)
    mape = stat.mean_absolute_percentage_error(y_output, y_pred)
    print('MAE = {:.3f} \n MAPE = {:.3f}%'.format(mae, mape))
    mse = stat.mean_squared_error(y_output, y_pred)
    rmse = stat.root_mean_squared_error(y_output, y_pred)
    print('MSE = {:.3f} \n RMSE = {:.3f}%'.format(mse, rmse))

    """
    Suppose your regression is y = W*x + b with x the scaled data, with the original data it is
    y = W/std * x0 + b - u/std * W
    where u and std are mean value and standard deviation of x0. Yet I don't think you need to transform back the data. Just use the same u and std to scale the new test data.
    """
    theta = np.zeros(2)
    

if __name__ == "__main__":
    main()

    """https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal"""