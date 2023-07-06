from train_model import LinearRegressionModel
from predict_price import PredictPrice


def main():
    try:
        my_model = LinearRegressionModel()
        print(my_model)
        my_model.plot_data()
    except:
        print("Error : cannot build model from training")
    try:
        predict_from_model = PredictPrice(my_model.get_coeffs)
    except (RuntimeError, TypeError, NameError):
        print("Error : cannot predict")


if __name__ == "__main__":
    """training model then predicting"""
    main()

# https://www.geeksforgeeks.org/gradient-descent-in-linear-regression/
