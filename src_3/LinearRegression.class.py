import numpy as np
import pandas as pd

class LinearRegressionGradientDescent:
    """ no normalization """
    def __init__(self, x_train, y_train):
        """ """
        self.x = x_train
        self.y = y_train
        return None
    
    def save_current_state(self, iter):
        """Calculate the model predictions: update_predicted_output
        np dot(X, theta) """
        self.predicted_y = self.theta[0] + self.x * self.theta[1]
        self.loss.append(self.cost)
        self.biases.append(self.theta[0])
        self.weights.append(self.theta[1])
        if (iter == 0 or iter == self.epochs - 1):
            print("Iteration = {}, Loss = {}".format(iter + 1, self.cost))
        else:
            print("Iteration = {}, Loss = {}".format(iter + 1, self.cost), end = '\r')

    def init_training(self, learning_rate, epochs):
        """ theta = [biais , weight] """
        self.alpha = learning_rate
        self.epochs = epochs
        self.theta = np.zeros(2)
        self.loss = []
        self.biases = []
        self.weights = []

    def predict_output(self):
        return self.theta[0] + self.x * self.theta[1]

    def train_gradient_descent(self, learning_rate = 0.5, epochs = 50):
        """ Initialize the model parameters
            Training loop :
                Calculate the model predictions
                Calculate the cost
                Use the gradient descent algorithm 
                Update the model parameters
        """
        self.init_training(learning_rate, epochs)
        for iter in range(epochs):
            predicted_y = self.theta[0] + self.x * self.theta[1]
            dy = predicted_y - self.y
            self.cost =  np.mean((dy) ** 2)
            self.save_current_state(iter)
            partial_derivative = np.zeros(2)
            partial_derivative[0] = 2 * np.mean(dy)
            partial_derivative[1] = 2 * np.mean(np.multiply(self.x, dy)) 
            self.theta -= self.alpha * partial_derivative
        #np.savetxt("./gradient_descent_model/theta.csv", self.theta, delimiter=",")

    """
    def denormalize_theta(self, y: np.ndarray) -> np.ndarray:
        rescaled_theta = np.zeros(2)
        rescaled_theta[0] = self.theta[0] * (np.max(y) - np.min(y)) + np.min(y)
        rescaled_theta[1] = np.sum(self.theta) - rescaled_theta[0] 
        np.savetxt("./gradient_descent_model/theta.csv", rescaled_theta, delimiter=",")
        print('Model to original dataset : \
            y = {:.3f} + x * {:.3f}'.format(rescaled_theta[0], rescaled_theta[1]))
        return rescaled_theta
    """
    def get_theta(self) -> np.ndarray:
        return self.theta
    
    def __str__(self):
        """ """
        return f'\x1b[6;33;46mTraining linear regression model using a gradient descent algorithm\
            \x1b[1;33;47m\nModel to Normalized dataset : \n \
            y = {self.theta[0]} + x * {self.theta[1]}\x1b[0m'

def mean_absolute_error(y, y_pred):
    absolute_error = abs(y - y_pred)
    mae = np.sum(absolute_error) / len(y)
    return mae

def mean_absolute_percentage_error(y, y_pred):
    """ It cannot be used if there are zero or close-to-zero values """
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

def denormalize_array(normarr: np.ndarray, y) -> np.ndarray:
    span = np.max(y) - np.min(y)
    return (normarr * span) + np.min(y)

def denormalize_element(element, y):
    span = np.max(y) - np.min(y)
    return (element * span) + np.min(y)

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
    x_train = normalize(arr[:,0])
    y_train = normalize(arr[:,1])
    normalized_model = LinearRegressionGradientDescent(x_train, y_train)
    normalized_model.train_gradient_descent()
    print(normalized_model)
    norm_theta = normalized_model.get_theta()
    # theta = model.denormalize_theta(y_output)
    y_pred_norm = normalized_model.predict_output()
    # y_pred_norm = predict_output(x_train, model.get_theta())
    y_pred = denormalize_array(y_pred_norm, y_output)
    theta = np.zeros(2)
    theta[0] = denormalize_element(norm_theta[0], y_output)
    theta[1] = (y_pred[-1] - y_pred[0]) / (x_input[-1] - x_input[0])
    np.savetxt("./gradient_descent_model/theta.csv", theta, delimiter=",")
    print(get_theta_from_csv(theta))
    # model accuracy
    mae = mean_absolute_error(y_output, y_pred)
    mape = mean_absolute_percentage_error(y_output, y_pred)
    print('MAE = {:.3f} \n MAPE = {:.3f}%'.format(mae, mape))
    mse = mean_squared_error(y_output, y_pred)
    rmse = root_mean_squared_error(y_output, y_pred)
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