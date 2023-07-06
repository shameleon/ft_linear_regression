import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearRegG4G:
    """copied from https://www.geeksforgeeks.org/ml-linear-regression/"""
    def __init__(self):
        self.parameters = {}
     
    def forward_propagation(self, train_input):
        m = self.parameters['m']
        c = self.parameters['c']
        predictions = np.multiply(m, train_input) + c
        # predictions = m * train_input + c
        return predictions
 
    def cost_function(self, predictions, train_output):
        cost = np.mean((train_output - predictions) ** 2)
        return cost
 
    def backward_propagation(self, train_input, train_output, predictions):
        derivatives = {}
        df = (train_output - predictions) * -1
        dm = np.mean(np.multiply(train_input, df))
        dc = np.mean(df)
        derivatives['dm'] = dm
        derivatives['dc'] = dc
        return derivatives
 
    def update_parameters(self, derivatives, learning_rate):
        self.parameters['m'] = self.parameters['m'] - learning_rate * derivatives['dm']
        self.parameters['c'] = self.parameters['c'] - learning_rate * derivatives['dc']
 
    def train(self, train_input, train_output, learning_rate, iters):
        #initialize random parameters
        self.parameters['m'] = 0
        self.parameters['c'] = 0
        self.loss = []
        for i in range(iters):
            #forward propagation
            predictions = self.forward_propagation(train_input)
            print(predictions)
            #cost function
            cost = self.cost_function(predictions, train_output)
            print(cost)
            #append loss and print
            self.loss.append(cost)
            print("Iteration = {}, Loss = {}".format(i+1, cost))
 
            #back propagation
            derivatives = self.backward_propagation(train_input, train_output, predictions)
 
            #update parameters
            self.update_parameters(derivatives, learning_rate)
 
        return self.parameters, self.loss

def main():
    # url = 'https://cdn.intra.42.fr/document/document/11434/data.csv'
    path = './data.csv'
    data = pd.read_csv(path, sep=",", usecols=['km', 'price'])
    data = data.dropna()
    # train_input = data['km'].to_numpy
    # train_output = data['price'].to_numpy
    train_input = np.array(data['km'])
    train_output = np.array(data['price'])
    print(train_output)
    linear_reg = LinearRegG4G()
    parameters, loss = linear_reg.train(train_input, train_output, 0.0005, 3)

    #Prediction on test data
    test_input = np.linspace(min(train_input), max(train_input), 20).astype(int)
    y_pred = test_input * parameters['m'] + parameters['c']
    
    # Plot the regression line with actual data pointa
    plt.plot(train_input, train_output, '+', label='Actual values')
    plt.plot(test_input, y_pred, label='Predicted values')
    plt.xlabel('Test input')
    plt.ylabel('Test Output or Predicted output')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
