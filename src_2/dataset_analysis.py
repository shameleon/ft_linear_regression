import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegressionModel:
    """ """
    def __init__(self, input, output, learning_rate = 0.05, epochs=1000):
        """ """
        self.input = input
        self.output = output
        self.origin = 0
        self.slope = 0
        self.learning_rate = learning_rate
        self.epochs = epochs
        return None

#     def get_coeffs(self):
#         coeffs = {"origin": self.origin, "slope": self.slope}
#         return coeffs

#     def __str__(self):
#         """https://www.scaler.com/topics/python-str/"""
#         return f'\x1b[6;30;60m Training [ok]\n model :\
#             y = {self.origin} + x * {self.slope}.\x1b[0m'

def correlation_coefficient(input: np.ndarray, output:np.ndarray) -> float:
    """ Pearson Product-Moment Correlation Coefficient.
    https://www.geeksforgeeks.org/python-pearson-correlation-test-between-two-variables/
    Measure the strength of the linear relationship between two quantitative variables.
    Parameters : input and output are numpy np.ndarray 1D-array of same shape (n, )
    Return : float whose value is within [-1, 1] interval
    """
    rho = np.corrcoef(input, output)[0, 1] #-0.85613942]
    print("Pearson correlation coefficient : {:.4f}".format(rho))
    result = ['very weak', 'weak', 'moderate', 'strong'] 
    strength = (abs(rho) >= 0.3) * 1 + (abs(rho) >= 0.5) * 1 + (abs(rho) >= 0.7) * 1
    print("Dataset has a {} linear correlation".format(result[strength]))
    return rho

def plot_dataset(df):
    plt.scatter('km','price' , c = 'blue', alpha = 0.5, data = df)
    plt.title('Car price according to mileage')
    plt.xlabel('mileage (km)')
    plt.ylabel('price ($)')
    plt.axis([0, 2.5e5, 0, 1e4])
    plt.grid(alpha = 0.5)
    plt.show()

def load_dataset(file: str):
    """ reads dataset from file or from url, into a pandas dataframe 
    returns two numpy arrays """
    url = 'https://cdn.intra.42.fr/document/document/18562/data.csv'
    try:
        df = pd.read_csv(file, sep=",", usecols=['km', 'price'])
        print('dataframe :', df.size())
    except:
        df = pd.read_csv(url, sep=",", usecols=['km', 'price'])
    df = df.dropna()
    print ("Dataset shape :", df.shape)
    return df

def main() -> None:
    """ """
    df = load_dataset(file = './datasets/data.csv')
    x_input = np.array(df['km'])
    y_output = np.array(df['price'])
    correlation_coefficient(x_input, y_output)
    plot_dataset(df)
    answer = input("Would you like to train a linear regression model (y / n) ? ")
    if (answer in ["y", "Y"]):
        print("Training dataset")
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
