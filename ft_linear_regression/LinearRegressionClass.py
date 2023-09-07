import numpy as np
from time import sleep
import plot_utils as plut
import statistics_utils as stat
import printout_utils as pout

"""LinearRegressionClass.py:

LinearRegressionGradientDescent class
performs more basic regression
"""

__author__ = "jmouaike"


class LinearRegressionGradientDescent:
    """ class LinearRegressionGradientDescent

        Model Hypothesis : estimated_y = θ0 + θ1.x
        θ0 = bias and θ0 = weight respectively origin and slope to the equation

        goal : obtain theta parameters [θ0, θ1]
        using Gradient Descent algorithm

        __init__
            parameters : training dataset as two separate 1D np arrays
        __save_current_state
            saves cost, bias and weight at each iteration
            cost printed to stdout at each iteration
            sleep for display rate
        predict_output
            predicted output is calculated with theta
        train_gradient_descent
            (re)initializes with specific learning_rate and epochs
            Training loop
                forward propagation
                backward propagation
            parameters : learning_rate, epochs
        plot_all_epochs
            subplots, dataset and loss, bias, weight plotted to epochs
        get_learning_params
        get_theta
        get_mean_error
        __str__
            return: equation as a string

        Static methods :
            - cost_function

        Parameters : training input, traing output as 1D numpy arrays
        Return : None
    """
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """ Parameters : training input, traing actual output"""
        self.x = x_train
        self.y = y_train
        pout.as_title('Training a linear regression model '
                      + 'using gradient descent algorithm')
        return None

    def __save_current_state(self, iter):
        """At each iteration of the training loop:
        - Loss, biases and weigths are appended to arrays
        for further analysis over epochs.
        - loss is displayed on stdout.
          sleep parameter might be changed to optimize display rate."""
        self.loss.append(self.cost)
        self.biases.append(self.theta[0])
        self.weights.append(self.theta[1])
        if (iter == 0 or iter == self.epochs - 1):
            print("Iteration = {}, Loss = {}".format(iter + 1, self.cost))
        else:
            print("Iteration = {}, Loss = {}".format(iter + 1, self.cost),
                  end='\r')
            sleep(0.0005)

    def __init_training(self, learning_rate, epochs):
        """ theta = [biais , weight] initialized """
        pout.as_status(f'alpha = {learning_rate}     epochs = {epochs}')
        self.alpha = learning_rate
        self.epochs = epochs
        self.theta = np.zeros(2)
        self.loss = []
        self.biases = []
        self.weights = []

    def predict_output(self):
        """ predicted output for a given theta and input (x) 1D array
        """
        return self.theta[0] + self.x * self.theta[1]

    def train_gradient_descent(self, learning_rate=0.05, epochs=1000):
        """ Initialize the model parameters
            Training loop :
                Calculate the model predictions
                Calculate the cost
                Use the gradient descent algorithm
                Update the model parameters
        """
        self.__init_training(learning_rate, epochs)
        for iter in range(epochs):
            predicted_y = self.predict_output()
            residual = predicted_y - self.y
            loss_elem = (residual) ** 2
            self.cost = np.sum(loss_elem) / (2 * len(residual))
            self.__save_current_state(iter)
            partial_derivative = np.zeros(2)
            partial_derivative[0] = np.mean(residual)
            partial_derivative[1] = np.mean(np.multiply(self.x, residual))
            self.theta -= self.alpha * partial_derivative
        self.y_pred = self.predict_output()

    def plot_all_epochs(self):
        """Subplots, dataset and loss, bias, weight plotted to epochs
        get_learning_params"""
        plut.plot_gradient_descent(self.x, self.y, self.y_pred,
                                   self.loss, self.biases, self.weights)

    def get_learning_params(self) -> str:
        return f'learning rate = {self.alpha}, iterations = {self.epochs}'

    def get_theta(self) -> np.ndarray:
        return self.theta

    def get_model_accuracy(self):
        stat.model_accuracy(self.y, self.y_pred)

    def __str__(self):
        """Returns model equation"""
        with np.printoptions(precision=3, suppress=True):
            equation = 'bias = {}, weight = {}'.format(self.theta[0],
                                                       self.theta[1])
        return f'\x1b[2;32;40m{equation}\x1b[0m'


def test_gradient_descent_class():
    """ Testing with another trained dataset """
    pout.as_title('TEST MODE : LinearRegressionGradientDescent class')
    print(LinearRegressionGradientDescent.__doc__)
    x_train = np.array([0.1, 0.3, 0.4, 0.8, 1])
    y_train = np.array([4, 2.5, 1.5, -1, -1.5])
    pout.as_title2('TEST : small dataset')
    test_model = LinearRegressionGradientDescent(x_train, y_train)
    for learning_rate in [0.01, 0.05, 0.1, 0.5]:
        test_model.train_gradient_descent(learning_rate, 100)
        print(test_model)
        test_model.get_model_accuracy()
        if pout.input_user_yes("Plot loss function over iterations"):
            test_model.plot_all_epochs()


if __name__ == "__main__":
    test_gradient_descent_class()
