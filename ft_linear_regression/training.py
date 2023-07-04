import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearRegressionModel:
    """ """
    def __init__(self):
        """ """
        self.url = 'https://cdn.intra.42.fr/document/document/11434/data.csv'
        self.data = pd.read_csv(self.url, sep=" ", usecols=['km', 'price'])
        self.x = self.data['km'].to_numpy
        self.y = self.data['price'].to_numpy
        self.arr = self.data[['km', 'price']].to_numpy()
        self.origin = 10000
        self.slope = -0.05
        return None

    def plot_data(self):
        """ """
        xn = np.linspace(min(self.arr[0]), max(self.arr[0]), 100).astype(int)
        yn = xn * self.slope + self.origin
        plt.scatter(self.data["km"], self.data["price"], c='orange')
        plt.plot(xn, yn)
        plt.show()
        return None

    def get_coeffs(self):
        coeffs = {"origin": self.origin, "slope": self.slope}
        return coeffs


"""
"""
