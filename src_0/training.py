import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearRegressionModel:
    """ """
    def __init__(self):
        """ """
        self.url = 'https://cdn.intra.42.fr/document/document/11434/data.csv'
        self.data = pd.read_csv(self.url, sep=",", usecols=['km', 'price'])
        #self.data.sort_values(by=['km'], inplace=True, ascending=True)
        self.x = self.data['km'].to_numpy
        self.y = self.data['price'].to_numpy
        self.arr = self.data[['km', 'price']].to_numpy()
        self.origin = 9000.0 #set to zero
        self.slope = -0.025 #set to zero
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

    def __str__(self):
        """https://www.scaler.com/topics/python-str/"""
        return f'\x1b[6;30;60m Training [ok]\n model : {self.origin} + m * {self.slope}.\x1b[0m'

"""
#https://docs.python.org/3/tutorial/errors.html

for arg in sys.argv[1:]:
    try:
        f = open(arg, 'r')
    except OSError:
        print('cannot open', arg)
    else:
        print(arg, 'has', len(f.readlines()), 'lines')
        f.close()

"""
