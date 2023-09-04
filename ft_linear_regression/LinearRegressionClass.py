import numpy as np
import pandas as pd
from my_colors import *
import statistics_utils as stat

class LinearRegressionGradientDescent:
    """ class LinearRegressionGradientDescent

        hypothesis : y = θ0 + θ1.x
        using GradientDescent algorithm
        goal : obtain theta parameters [θ0, θ1] 
            θ0 = bias and θ0 = weight respectively origin and slope to the equation

        self.__init__ 
            parameters : training dataset as two separate 1D np arrays
        self.train_gradient_descent
            Training loop 
                forward propagation
            parameters : learning_rate, epochs
        self.__str__
            return: equation as a string
        self.get_theta
            return theta parameters
    """
    def __init__(self, x_train:np.ndarray, y_train:np.ndarray):
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

    def get_theta(self) -> np.ndarray:
        return self.theta
    
    def __str__(self):
        """ """
        return f'\x1b[6;33;46mTraining linear regression model using a gradient descent algorithm\
            \x1b[1;33;47m\nModel to dataset : \n \
            y = {self.theta[0]} + x * {self.theta[1]}\x1b[0m'

def test_gradient_descent_class():
    """ """
    print(f'\n{COL_BLUWHI}----------- TEST MODE : LinearRegressionGradientDescent class -----------{COL_RESET}\n')
    print(LinearRegressionGradientDescent.__doc__)
    x_train = np.array([0.1, 0.3, 0.4, 0.8, 1])
    y_train = np.array([ 4, 2.5, 1.5, -1, -1.5  ])
    print(f'\n{COL_BLUCYA}----------- TEST : small dataset -----------{COL_RESET}\n')
    test_model = LinearRegressionGradientDescent(x_train, y_train)
    test_model.train_gradient_descent(0.05, 100)
    print(test_model)

if __name__ == "__main__":
    test_gradient_descent_class()