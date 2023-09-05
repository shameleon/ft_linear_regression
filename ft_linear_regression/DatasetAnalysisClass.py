import numpy as np
import pandas as pd
import os
from LinearRegressionClass import LinearRegressionGradientDescent
import plot_utils as plut
import statistics_utils as stat
from color_out import *

class CarPriceDatasetAnalysis:
    """
    class CarPriceDatasetAnalysis :
        loads dataset from a file or url
        asks for preview analysis (plot and some basic statistics)
        perfoms dataset normalization
        instantiates LinearRegressionGradientDescent class 
        obtains theta parameters from the latest and export it to a file
    """
    def __init__(self,
                 source = f'data.csv',
                 dest = "theta.csv",
                 normalize = True) -> None:
        self.source_file = source
        self.dest_file = dest
        self.dest_path = "./gradient_descent_model/"
        self.__load_dataset()
        self.normalize = normalize
        self.__normalize_dataset()
        print_check('Dataset ready for training')
        return None
    
    def __load_dataset(self):
        try:
            self.df = pd.read_csv(self.source_file)
        except:
            print_stderr('Error: could not open file. No data, no model to train.')
            exit(0)
        self.df = self.df.dropna()
        arr = self.df.to_numpy()
        self.x_input = arr[:,0]
        self.y_output = arr[:,1]
        if len(self.x_input) < 2 or len(self.y_output) < 2:
            print_stderr('Error: datapoints missing, no model to train.')
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
    
    def estimated_output_error(self):
        """  """
        y_denorm = stat.denormalize_array(self.gradient_model.predict_output(), self.y_output)
        y_estimated = self.theta[0] + self.x_input * self.theta[1]
        print(y_denorm - y_estimated)


    def __denormalize_and_save_theta(self):
        if self.normalize:
            print("Normalized dataset : ",self.gradient_model)
            norm_theta = self.gradient_model.get_theta()
            y_pred_norm = self.gradient_model.predict_output()
            self.y_pred = stat.denormalize_array(y_pred_norm, self.y_output)
            theta = np.ones(2)
            theta[1] = (self.y_pred[-1] - self.y_pred[0]) / (self.x_input[-1] - self.x_input[0])
            theta[0] = self.y_pred[0] - self.x_input[0] * theta[1]
        else:
            print("Dataset : ",self.gradient_model)
            theta = self.gradient_model.get_theta()
        if not os.path.exists(self.dest_path):
            os.makedirs(self.dest_path)
        np.savetxt(f'{self.dest_path + self.dest_file}', theta, delimiter=",")
        print_result(f'Linear regression model equation to dataset : \n \
            estimated_price = {theta[0]} + ({theta[1]}) * mileage'.format(theta[0], theta[1]))
        self.theta = theta


    def train_dataset(self, learning_rate = 0.05, epochs = 100) -> None:
        """ Instanciate GradientDescent class then train model with training dataset.
        if needed, denormalizes results to obtain theta from unnormalized dataset.
        Saves theta to file.
        Print equation with parameters obtained from gradient descent model.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        if input_user_yes('Preview analysis for dataset'):
            print_title3('Dataset preview')
            self.dataset_preview()
        if not input_user_yes("Linear regression training with gradient descent algorithm"):
            # end training prog
            return None
        self.gradient_model = LinearRegressionGradientDescent(self.x_train, self.y_train)
        self.gradient_model.train_gradient_descent(learning_rate, epochs)
        self.__denormalize_and_save_theta()
        self.__post_training_analysis()
        return None
    
    def __post_training_analysis(self):
        """ asks to perform statistic analysis with dataset output and trained model parameters
            ask to plot loss and parameters to epochs during the run of gradient descent algorithm
            ask to draw the final plot : dataset with the regression line
        """
        if input_user_yes("Model accuracy statistics") == True:
            stat.model_accuracy(self.y_output, self.y_pred, self.theta)
        if input_user_yes("Plot loss function and parameters over iteration epochs"):
            print_title2("Trained " + self.normalize * "normalized " + "dataset")
            print_title3("Subplots for gradient descent algorithm")
            self.gradient_model.plot_all_epochs()
        if input_user_yes("Draw Final Plot : model regression line to trained dataset"):
            print_title("Trained dataset")
            print_title3("Subplots for gradient descent algorithm")
            self.plot_final()

    def plot_final(self):
        y_pred = self.theta[0] + self.x_input * self.theta[1]
        suptitle = ('ft_linear regression : gradient descent algorithm')
        title = self.gradient_model.get_learning_params()
        # title += self.__str__()
        plut.plot_final(self.x_input, self.y_output, y_pred, suptitle, title)

    def __str__(self):
        with np.printoptions(precision=3, suppress=True):
            equation = 'price = {} + {}.mileage'.format(self.theta[0], self.theta[1])
        return f'\x1b[2;32;40m{equation}\x1b[0m'


def test_dataset_analysis_class() -> None:
    """ """
    test_model = CarPriceDatasetAnalysis()
    test_model.train_dataset(0.2, 500)
    test_model.estimated_output_error()
    return None

if __name__ == "__main__":
    """training model"""
    test_dataset_analysis_class()