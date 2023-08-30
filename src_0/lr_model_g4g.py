import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

class LinearRegG4G:
    """copied from https://www.geeksforgeeks.org/ml-linear-regression/"""
    def __init__(self):
        self.parameters = {}
     
    def forward_propagation(self, train_input):
        m = self.parameters['m']
        c = self.parameters['c']
        # print('\x1b[6;30;33m m = {}   c = {}\x1b[0m\n'.format(m, c))
        predictions = np.multiply(m, train_input) + c
        # predictions = m * train_input + c
        return predictions
 
    def cost_function(self, predictions, train_output):
        cost = np.mean((train_output - predictions) ** 2)
        return cost
 
    def backward_propagation(self, train_input, train_output, predictions):
        derivatives = {}
        df = predictions - train_output
        dm = 2 * np.mean(np.multiply(train_input, df))
        dc = 2 * np.mean(df)
        derivatives['dm'] = dm
        derivatives['dc'] = dc
        # print('dm = {}  dc = {}'.format(dm, dc))
        return derivatives
 
    def update_parameters(self, derivatives, learning_rate):
        self.parameters['m'] -= learning_rate * derivatives['dm']
        self.parameters['c'] -= learning_rate * derivatives['dc']
 
    def train(self, train_input, train_output, learning_rate, iters=50):
        #initialize random parameters
        self.parameters['m'] = 0
        self.parameters['c'] = 0
        self.loss = []
        for i in range(iters):
            # forward propagation
            predictions = self.forward_propagation(train_input)
            # print(predictions)
            # cost function
            cost = self.cost_function(predictions, train_output)
            #append loss and print
            self.loss.append(cost)
            print("Iteration = {}, Loss = {}".format(i+1, cost))
 
            # back propagation
            derivatives = self.backward_propagation(train_input, train_output, predictions)
 
            # update parameters
            self.update_parameters(derivatives, learning_rate)
 
        return self.parameters, self.loss

def normalize(arr):
    min = np.min(arr)
    max = np.max(arr)
    range = max - min
    return (arr - min) / range

def main():
    # url = 'https://cdn.intra.42.fr/document/document/11434/data.csv'
    learning_rate = 0.05
    epochs = 1000
    path = './data/data.csv'
    data = pd.read_csv(path, sep=",", usecols=['km', 'price'])
    data = data.dropna()
    # train_input = data['km'].to_numpy
    # train_output = data['price'].to_numpy
    x_input = np.array(data['km'])
    train_input = normalize(x_input)
    y_output = np.array(data['price'])
    train_output = normalize(y_output)
    linear_reg = LinearRegG4G()
    parameters, loss = linear_reg.train(train_input, train_output, learning_rate, epochs)

    #Prediction on test data
    test_input = np.linspace(min(train_input), max(train_input), 20)
    y_pred = test_input * parameters['m'] + parameters['c']

    #mse = np.square(np.subtract(y_output,y_pred)).mean() 
    #rmse = math.sqrt(MSE)
    # Plot the regression line with actual data pointa
    """
    plt.plot(train_input, train_output, '+', label='Actual values')
    plt.plot(test_input, y_pred, label='Predicted values')
    plt.xlabel('Test input')
    plt.ylabel('Test Output or Predicted output')
    plt.legend()
    plt.show()
    plt.plot(loss)
    plt.show()
    """
    # equation = f'y = {.2f} x +{.2f}'.format(parameters['m'], parameters['c'])
    equation = 'Hello'
    fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(24, 8))
    i = 0
    ax[0].text(0, 0.5, equation, color="green", fontsize=18, ha='center')
    ax[0].plot(x_input, y_output, 'o', color="green", label='price')
    ax[0].set_xlabel('mileage (km)')
    ax[0].set_ylabel('price ($)')
    ax[0].legend()
    i += 1
    ax[i].plot(train_input, train_output, '+', label='Actual values')
    ax[i].plot(test_input, y_pred, label='Predicted values')
    ax[i].set_xlabel('normalized input')
    ax[i].set_ylabel('normalized output')
    ax[i].legend()
    ax[i].grid(True)
    i +=1
    ax[i].plot(loss)
    ax[i].set_title('Loss function')
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel('normalized output')
    i += 1
    y_residual = np.subtract(train_output, train_input * parameters['m'] + parameters['c'])
    ax[i].plot(train_input, abs(y_residual), 'x', color="red", label='residual')
    plt.show()
    # https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html
    # https://matplotlib.org/stable/tutorials/text/index.html

if __name__ == "__main__":
    main()
