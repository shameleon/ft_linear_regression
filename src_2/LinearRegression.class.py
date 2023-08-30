import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression:
    """ """
    def __init__(self, learning_rate = 0.05, epochs=1000):
        """ """
        url = 'https://cdn.intra.42.fr/document/document/18562/data.csv'
        # path = './data/data.csv'
        data = pd.read_csv(url, sep=",", usecols=['km', 'price'])
        # print(data.size())
        data = data.dropna()
        self.input = data['km'].to_numpy
        self.output = data['price'].to_numpy
        # self.arr = self.data[['km', 'price']].to_numpy()
        self.origin = 0    # set to zero
        self.slope = 0     # set to zero
        self.learning_rate = learning_rate
        self.epochs = epochs
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
        return f'\x1b[6;30;60m Training [ok]\n model :\
            y = {self.origin} + x * {self.slope}.\x1b[0m'

def main() -> None:
    """ """
    load_dataset(path = './datasets/data.csv')
    my_model = LinearRegressionModel(0.05, 100)
    print("Model training")
    print(my_model)
    return None

if __name__ == "__main__":
    """training model"""
    main()

"""
https://www.geeksforgeeks.org/gradient-descent-in-linear-regression/

https://docs.python.org/3/tutorial/errors.html

for arg in sys.argv[1:]:
    try:
        f = open(arg, 'r')
    except OSError:
        print('cannot open', arg)
    else:
        print(arg, 'has', len(f.readlines()), 'lines')
        f.close()

"""
