import numpy as np
import pandas as pd
import os
from LinearRegressionClass import LinearRegressionGradientDescent
import plot_utils as plut
import statistics_utils as stat
import printout_utils as pout

"""DatasetAnalysisClass.py:

CarPriceDatasetAnalysis class

performs Linear Regression with gradient descent
with many additional features :
    dataset safeguards
    normalization, denormalization
    verbose mode
    plots for dataset and model
    plots for loss function
    plots for gradient descent
    data persistence for model parameters
    a model accuracy report to measure the model error
"""

__author__ = "jmouaike"


class CarPriceDatasetAnalysis:
    """
    class CarPriceDatasetAnalysis :
        loads dataset from a file or url
        asks for preview analysis (plot and some basic statistics)
        perfoms dataset normalization
        instantiates LinearRegressionGradientDescent class
        obtains theta parameters from the latest and export it to a file
    """
    def __init__(self, source='data.csv', dest="theta.csv",
                 bonus=True) -> None:
        self.source_file = source
        self.dest_file = dest
        self.dest_path = "./gradient_descent_model/"
        self.__load_dataset()
        self.normalize = True
        self.__normalize_dataset()
        self.bonus = bonus
        pout.as_check('Dataset ready for training')
        return None

    def __load_dataset(self):
        """ Reads dataset file to numpy 1D arrays.
            Quality control to the file and the dataset
            Exception is raised if any problem with the file
        """
        try:
            self.df = pd.read_csv(self.source_file)
        except (FileNotFoundError, ValueError,
                TypeError, IndexError, AttributeError) as e:
            pout.as_error(f'Error: failed to load {self.source_file} file.')
            print("\t", *e.args)
            pout.as_cross("\tNo data, no model to train.")
            pout.as_comment("END :(")
            exit(0)
        self.df = self.df.dropna()
        arr = self.df.to_numpy()
        self.x_input = arr[:, 0]
        self.y_output = arr[:, 1]
        if len(self.x_input) < 2 or len(self.y_output) < 2:
            pout.as_error('Error: datapoints missing, no model to train.')
            exit(0)

    def __normalize_dataset(self):
        """ normalize dataset if boolean self.normalize
            exception is raised if any type issue with numpy arrays
        """
        if self.normalize:
            try:
                self.x_train = stat.normalize(self.x_input)
                self.y_train = stat.normalize(self.y_output)
            except TypeError as e:
                pout.as_error("Error: Type error")
                print("\t", *e.args)
                pout.as_comment("END :(")
                exit(0)
        else:
            self.x_train = self.x_input
            self.y_train = self.y_input

    def dataset_preview(self):
        stat.correlation_coefficient(self.x_input, self.y_output)
        if pout.input_user_yes("Plot the dataset"):
            plut.plot_dataset(self.df)
        if pout.input_user_yes("Statistical linear_regression model analysis"
                               + "for dataset"):
            statistic_model = stat.StatisticLinearRegression(self.x_input,
                                                             self.y_output)
            statistic_model.calculate_params()
        if pout.input_user_yes("Plot the cost function"):
            plut.plot_cost_function(self.x_train, self.y_train)

    def train_dataset(self, learning_rate=0.05, epochs=100) -> None:
        """ Instanciate GradientDescent class then train model
        with training dataset.
        if needed, denormalizes results to obtain theta
        from unnormalized dataset.
        Saves theta to file.
        Print equation with parameters obtained from gradient descent model.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        if self.bonus:
            if pout.input_user_yes('Preview analysis for dataset'):
                pout.as_title3('Dataset preview')
                self.dataset_preview()
            if not pout.input_user_yes("Linear regression training "
                                       + "with gradient descent algorithm"):
                return None
        self.gradient_model = LinearRegressionGradientDescent(self.x_train,
                                                              self.y_train)
        self.gradient_model.train_gradient_descent(learning_rate, epochs)
        self.__denormalize_and_save_theta()
        pout.as_check("Model trained")
        if self.bonus:
            self.__post_training_analysis()
        else:
            mean_err = stat.mean_error(self.y_output, self.y_pred)
            pout.as_title2('Mean_error = {:.4f}'.format(mean_err))
        return None

    def __denormalize_and_save_theta(self):
        """ After training, the resulting Theta for original dataset
        is obtained from predicted output (y_pred). then saved to file"""
        if self.normalize:
            print("Normalized dataset : ", self.gradient_model)
            y_pred_norm = self.gradient_model.predict_output()
            self.y_pred = stat.denormalize_array(y_pred_norm, self.y_output)
            theta = np.ones(2)
            delta_y = self.y_pred[-1] - self.y_pred[0]
            delta_x = self.x_input[-1] - self.x_input[0]
            theta[1] = delta_y / delta_x
            theta[0] = self.y_pred[0] - self.x_input[0] * theta[1]
        else:
            print("Dataset : ", self.gradient_model)
            theta = self.gradient_model.get_theta()
        if not os.path.exists(self.dest_path):
            os.makedirs(self.dest_path)
        np.savetxt(f'{self.dest_path + self.dest_file}', theta, delimiter=",")
        pout.as_result("Linear regression model equation to dataset :")
        pout.as_result('\testimated_price = {:4f}'.format(theta[0])
                       + ' + ({:4f}) * mileage'.format(theta[1]))
        self.theta = theta

    def __post_training_analysis(self):
        """
        - Performs statistic analysis with dataset output
        and trained model parameters.
        - Plots loss and parameters to epochs during the run of
        gradient descent algorithm.
        - Draw the final plot : dataset with the regression line.
        """
        if pout.input_user_yes("Model accuracy statistics"):
            stat.model_accuracy(self.y_output, self.y_pred)
        if pout.input_user_yes("Plot loss function and parameters"
                               + "over iteration epochs"):
            pout.as_title2("Trained " + self.normalize * "normalized "
                           + "dataset")
            pout.as_check("Subplots for gradient descent algorithm")
            self.gradient_model.plot_all_epochs()
        if pout.input_user_yes("Draw Final Plot : model regression line"
                               + "to trained dataset"):
            pout.as_title2("Trained dataset")
            pout.as_check("Final plot for linear regression"
                          + "with gradient descent algorithm")
            self.plot_final()
        pout.as_comment("END :)")

    def plot_final(self):
        """ final plot represent dataset with reegression line
        and a second panel showing residual output """
        y_pred = self.theta[0] + self.x_input * self.theta[1]
        suptitle = ('ft_linear regression : gradient descent algorithm')
        title = self.gradient_model.get_learning_params()
        title += self.__str__()
        plut.plot_final(self.x_input, self.y_output, y_pred, suptitle, title)

    def __str__(self):
        """ Equation hypothesis to the linear regression model """
        equation = '\nestimated_price = {:.4f}'.format(self.theta[0])
        equation += ' + ({:.4f}) * mileage'.format(self.theta[1])
        return equation


def test_dataset_analysis_class() -> None:
    """ train a model with different learning rates epochs
    bonus = false will deactivate options such as plots and most stats"""
    print(CarPriceDatasetAnalysis.__doc__)
    test_model = CarPriceDatasetAnalysis(bonus=False)
    for learning_rate in [0.02, 0.1, 0.5]:
        for epochs in [100, 500, 2000]:
            test_model.train_dataset(learning_rate, epochs)
    return None


if __name__ == "__main__":
    """training model"""
    test_dataset_analysis_class()
