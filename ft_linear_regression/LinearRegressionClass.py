import numpy as np
from color_out import *
import plot_utils as plut
from statistics_utils import mean_error, model_accuracy
from time import sleep

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
            displays cost at each iteration
            sleep in comment
        
        predict_output
            predicted output is calculated with theta.[1 x]
        
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
    """
    def __init__(self, x_train:np.ndarray, y_train:np.ndarray) -> None:
        """ """
        self.x = x_train
        self.y = y_train
        print_title('Training a linear regression model using gradient descent algorithm')
        return None
    
    def __save_current_state(self, iter):
        """Calculate the model predictions: update_predicted_output
        np dot(X, theta) """
        self.predicted_y = self.theta[0] + self.x * self.theta[1]
        self.loss.append(self.cost)
        self.biases.append(self.theta[0])
        self.weights.append(self.theta[1])
        if (iter == 0 or iter == self.epochs - 1):
            print("Iteration = {}, \tLoss = {}".format(iter + 1, self.cost))
        else:
            print("Iteration = {}, \tLoss = {}".format(iter + 1, self.cost), end = '\r')
            # sleep(0.01)

    def __init_training(self, learning_rate, epochs):
        """ theta = [biais , weight] """
        print_status(f'alpha = {learning_rate}     epochs = {epochs}')
        self.alpha = learning_rate
        self.epochs = epochs
        self.theta = np.zeros(2)
        self.loss = []
        self.biases = []
        self.weights = []

    def predict_output(self):
        return self.theta[0] + self.x * self.theta[1]

    def train_gradient_descent(self, learning_rate = 0.05, epochs = 1000):
        """ Initialize the model parameters
            Training loop :
                Calculate the model predictions
                Calculate the cost
                Use the gradient descent algorithm 
                Update the model parameters
        """
        self.__init_training(learning_rate, epochs)
        for iter in range(epochs):
            predicted_y = self.theta[0] + self.x * self.theta[1]
            residual = predicted_y - self.y
            loss_elem = (residual) ** 2
            self.cost = np.sum(loss_elem) / ( 2 * len(residual))
            #self.cost =  np.mean((dy) ** 2)
            self.__save_current_state(iter)
            partial_derivative = np.zeros(2)
            partial_derivative[0] = 2 * np.mean(residual)
            partial_derivative[1] = 2 * np.mean(np.multiply(self.x, residual)) 
            self.theta -= self.alpha * partial_derivative

    def plot_all_epochs(self):
        y_pred = self.theta[0] + self.x * self.theta[1]
        plut.plot_gradient_descent(self.x, self.y, y_pred, \
                                    self.loss, self.biases, self.weights)

    def get_learning_params(self) -> str:
        return  f'learning rate = {self.alpha}, iterations = {self.epochs}'

    def get_theta(self) -> np.ndarray:
        return self.theta
    
    def get_mean_error(self) -> float:
        y_pred = self.theta[0] + self.x * self.theta[1]
        me = mean_error(self.y, y_pred)
        return me
    
    def get_model_accuracy(self):
        y_pred = self.theta[0] + self.x * self.theta[1]
        model_accuracy(self.y, y_pred, self.theta)
    
    def __str__(self):
        """ """
        with np.printoptions(precision=3, suppress=True):
            equation = 'bias = {}, weight = {}'.format(self.theta[0], self.theta[1])
        return f'\x1b[2;32;40m{equation}\x1b[0m'
        # return f'\x1b[2;32;40my = {self.theta[0]} + x * {self.theta[1]}\x1b[0m'

def test_gradient_descent_class():
    """ """
    print_title(f'TEST MODE : LinearRegressionGradientDescent class')
    print(LinearRegressionGradientDescent.__doc__)
    x_train = np.array([0.1, 0.3, 0.4, 0.8, 1])
    y_train = np.array([ 4, 2.5, 1.5, -1, -1.5  ])
    print_title2('TEST : small dataset')
    test_model = LinearRegressionGradientDescent(x_train, y_train)
    for learning_rate in [0.01, 0.05, 0.1, 0.5]:
        test_model.train_gradient_descent(learning_rate, 100)
        print(test_model)
        # print_result(f'Mean Error = {test_model.get_mean_error()}')
        test_model.get_model_accuracy()
        if input_user_yes("Plot loss function over iterations"):
            test_model.plot_all_epochs()

if __name__ == "__main__":
    test_gradient_descent_class()