import numpy as np
import pandas as pd
import plot_utils as plut
import statistics_utils as stat
from my_colors import *

class CarPriceDatasetAnalysis:
    """
    class DatasetAnalysis :
        loads dataset from a file or url
        asks for preview analysis (plot and some basic statistics)
        perfoms dataset normalization
        instantiates LinearRegressionGradientDescent class 
        obtains theta parameters from the latest and export it to a file
    """
    def __init__(self,
                 source = f'data.csv',
                 dest = "./gradient_descent_model/theta.csv",
                 normalize = True) -> None:
        self.source_file = source
        self.dest_file = dest 
        self.__load_dataset()
        self.normalize = normalize
        self.__normalize_dataset()
        return None
    
    def __load_dataset(self):
        try:
            self.df = pd.read_csv(self.source_file)
        except:
            print('Error: could not open file. No data, no model to train.')
            exit(0)
        self.df = self.df.dropna()
        arr = self.df.to_numpy()
        self.x_input = arr[:,0]
        self.y_output = arr[:,1]

    def __normalize_dataset(self):
        if self.normalize:
            self.x_train = stat.normalize(self.x_input)
            self.y_train = stat.normalize(self.y_output)
        else:
            self.x_train = self.x_input
            self.y_train = self.y_input
    
    def dataset_preview(self):
        stat.correlation_coefficient(self.x_input, self.y_output)
        answer = input("Plot the dataset (y / n) ? ")
        if (answer in ["y", "Y"]):
            plut.plot_dataset(self.df)
        answer = input("Statistical linear_regression model analysis for dataset (y / n) ? ")
        if (answer in ["y", "Y"]):
            statistic_model = stat.StatisticLinearRegression(self.x_input, self.y_output)
        answer = input("Plot the cost function (y / n) ? ")
        if (answer in ["y", "Y"]):
            plut.plot_cost_function(self.x_train, self.y_train)


def test_dataset_analysis_class() -> None:
    """ """
    test_model = CarPriceDatasetAnalysis()
    answer = input("Would you like a preview analysis for dataset (y / n) ? ")
    if (answer in ["y", "Y"]):
        print("------------- dataset preview -------------")
        test_model.dataset_preview()
    return None

if __name__ == "__main__":
    """training model"""
    test_dataset_analysis_class()