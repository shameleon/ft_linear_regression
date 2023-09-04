import numpy as np
import pandas as pd
from LinearRegressionClass import LinearRegressionGradientDescent
import plot_utils as plut
import statistics_utils as stat
from color_out import *

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
        print_check('dataset ready for training')
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
        if len(self.x_input) < 2 or len(self.y_output) < 2:
            print('Error: datapoints missing, no model to train.')
            exit(0)

    def __normalize_dataset(self):
        if self.normalize:
            self.x_train = stat.normalize(self.x_input)
            self.y_train = stat.normalize(self.y_output)
        else:
            self.x_train = self.x_input
            self.y_train = self.y_input
    
    def dataset_preview(self):
        stat.correlation_coefficient(self.x_input, self.y_output)
        if input_user_yes("Plot the dataset"):
            plut.plot_dataset(self.df)
        if input_user_yes("Statistical linear_regression model analysis for dataset"):
            statistic_model = stat.StatisticLinearRegression(self.x_input, self.y_output)
        if input_user_yes("Plot the cost function"):
            plut.plot_cost_function(self.x_train, self.y_train)

    def train_dataset(self) -> None:
        if not input_user_yes("Linear regression training with gradient descent algorithm"):
            # end training prog
            return None
        self.gradient_model = LinearRegressionGradientDescent(self.x_train, self.y_train)
        self.gradient_model.train_gradient_descent()
        print("Normalized dataset : ",self.gradient_model)
        norm_theta = self.gradient_model.get_theta()
        y_pred_norm = self.gradient_model.predict_output()
        self.y_pred = stat.denormalize_array(y_pred_norm, self.y_output)
        theta = np.zeros(2)
        theta[0] = stat.denormalize_element(norm_theta[0], self.y_output)
        theta[1] = (self.y_pred[-1] - self.y_pred[0]) / (self.x_input[-1] - self.x_input[0])
        np.savetxt(self.dest_file, theta, delimiter=",")
        print_result(f'Model equation to dataset : \n \
            y = {theta[0]} + x * {theta[1]}'.format(theta[0], theta[1]))
        self.theta = theta
        if input_user_yes("Model accuracy statistics") == True:
            stat.model_accuracy(self.y_output, self.y_pred, self.theta)
        if input_user_yes("Plot loss function over iterations"):
            print_title3("Normalize dataset, subplots for gradient descent algorithm")
            self.gradient_model.plot_all()


def test_dataset_analysis_class() -> None:
    """ """
    test_model = CarPriceDatasetAnalysis()
    if input_user_yes('Preview analysis for dataset'):
        print_title3('dataset preview')
        test_model.dataset_preview()
    test_model.train_dataset()
    return None

if __name__ == "__main__":
    """training model"""
    test_dataset_analysis_class()