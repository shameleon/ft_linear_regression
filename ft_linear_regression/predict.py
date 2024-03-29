import printout_utils as pout
from PredictPriceClass import PredictPriceFromModel

"""predict.py:

Program to predict a car price from a user-given mileage

"""

__author__ = "jmouaike"


def main() -> None:
    """ Predicting car price program.
    Asks for a car mileage in km and predicts the car's price.
    Price is calculated by PredictPriceFromModel class instance
    and is based on linear regression parameters
    that are eventually found in the source file.
    """
    source_file = "./gradient_descent_model/theta.csv"
    pout.as_title('PREDICT A CAR PRICE')
    model_prediction = PredictPriceFromModel(source_file)
    while True:
        model_prediction.ask_for_mileage()
        if not pout.input_user_yes('Continue, for another price prediction'):
            break
    pout.as_comment("END :)")
    return None


if __name__ == "__main__":
    main()
