import numpy as np
from color_out import *
from PredictPriceClass import PredictPriceFromModel

def main() -> None:
    """ Predicting car price program.
    Asks for a car mileage in km and predicts the car's price.
    Price is calculated by PredictPriceFromModel class instance
    and is based on linear regression parameters
    that are eventually found in the source file.
    """
    source_file = "./gradient_descent_model/theta.csv"
    print_title('PREDICT A CAR PRICE')
    model_prediction = PredictPriceFromModel(source_file)
    while True:
        model_prediction.ask_for_mileage()
        if not input_user_yes('Continue, for another price prediction'):
            break
    print_comment("END :)")
    return None

if __name__ == "__main__":
    main()
