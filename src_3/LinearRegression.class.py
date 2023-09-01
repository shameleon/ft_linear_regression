import numpy as np
import pandas as pd

class LinearRegression:
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

    def train_gradient_descent(self, learning_rate = 0.1, epochs = 50):
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
            derivative_weight = 2 * np.mean(np.multiply(self.x, dy)) 
            derivative_bias = 2 * np.mean(dy)
            self.theta[1] -= self.alpha * derivative_weight
            self.theta[0] -= self.alpha* derivative_bias
        np.savetxt("./gradient_descent_model/theta.csv", self.theta, delimiter=",")

    def __str__(self):
        """ """
        return f'\x1b[6;30;60m Training linear regression model using a gradient descent algorithm :\
            y = {self.theta[0]} + x * {self.theta[1]}.\x1b[0m'


def mean_absolute_error(y, y_pred):
    absolute_error = abs(y - y_pred)
    mae = np.sum(absolute_error) / len(y)
    return mae

def mean_absolute_percentage_error(y, y_pred):
    mean_absolute_error = abs((y - y_pred) / y) / len(y)
    mape =  np.sum(mean_absolute_error) * 100 
    return mape

def main():
    df = pd.read_csv(f'data.csv')
    labels = df.keys()
    arr = df.to_numpy()
    x_train = arr[:,0]
    y_train = arr[:,1]
    model = LinearRegression(x_train, y_train)
    model.train_gradient_descent()
    y_pred = model.predict_output()
    mae = mean_absolute_error(y_train, y_pred)
    mape = mean_absolute_percentage_error(y_train, y_pred)
    print(model)
    print('MAE = {:.3f} \t MAPE = {:.3f}%'.format(mae, mape))

if __name__ == "__main__":
    main()