import numpy as np
import printout_utils as pout


class PredictPriceFromModel():
    """ class for predicting price for a given mileage
        based on linear regression parameters, theta = [θ0, θ1]

        hypothesis : price = θ0 + θ1 * mileage

        self.__init__        : takes file name as parameter
        self.__upload_model  : loads parameters from file
        self.ask_for_mileage : user enters a mileage
                             : will raise exceptions
        self.__predict_price : calculates the price based on theta
        self.__str__         : returns prediction result as a string

        Nested classes for Exceptions : out range mileages and prices
    """
    def __init__(self, file: str) -> None:
        """parameter : file that contains parameters created by training.py """
        self.model_file = file
        self.theta = np.zeros(2)
        self.__upload_model()
        self.mileage = 0
        self.price = self.__predict_price(self.mileage)
        return None

    def __upload_model(self):
        try:
            self.theta = np.loadtxt(self.model_file)
            pout.as_check('Linear regression model parameters loaded')
        except (FileNotFoundError, ValueError, IndexError):
            pout.as_cross('Linear regression model parameters not found\
                          : initialized to default')
        color = ['\x1b[0m', '\x1b[1;32;40m']
        print(f'{color[1]}\thypothesis :\n\tprice = θ0 + θ1 * mileage')
        print('\tθ0 = {:.2f}     \
              θ1 = {:6f}'.format(self.theta[0], self.theta[1]))
        print(f'{color[0]}')

    def __predict_price(self, mileage: float) -> float:
        """
        Estimated price is calculated with dot product :
        estimated price = [θ0, θ1].[    1    ]
                                   [ mileage ]
        """
        vec = np.ones(2)
        vec[1] = mileage
        price = np.dot(self.theta, vec)
        return price

    def ask_for_mileage(self):
        """ """
        try:
            in_str = pout.input_user_str('Please, enter a car mileage (km) :')
            self.mileage = float(in_str)
            if self.mileage < 0 or self.mileage > 1E6:
                raise InvalidMileageRangeError(self.mileage)
            self.price = self.__predict_price(self.mileage)
            if self.price < 0:
                raise NegativePredictedPriceError()
        except (RecursionError, RuntimeError, TypeError, ValueError):
            pout.as_error("Error : input is not valid")
        except (EOFError):
            pout.as_error("Error : unexpected end of file")
        except (NegativePredictedPriceError):
            self.price = 0
            print("Mileage too High: predicted price is set to zero")
            print(self)
        except (InvalidMileageRangeError):
            pout.as_error("Error: mileage is out of range")
        else:
            print(self)
        return None

    def __str__(self):
        return f'mileage = {format(self.mileage, ".0f")} km\
            predicted price = $ {format(self.price, ".2f")}\n'


class InvalidMileageRangeError(Exception):
    """ Exception raised for errors in the input Mileage """
    def __init__(self, mileage, message="Invalid mileage : out of range"):
        self.mileage = mileage
        self.message = message
        super().__init__(self.message)


class NegativePredictedPriceError(Exception):
    """ Exception raised for errors in the input Mileage """

    def __init__(self, message="Invalid price : out of range"):
        self.message = message
        super().__init__(self.message)


def test_predict_price_class(theta_file: str):
    """ tests for PredictPriceFromModel class """
    pout.as_title2('TEST MODE : PredictPriceFromModel class')
    print(PredictPriceFromModel.__doc__)
    pout.as_title3('TEST1 : no model')
    price_no_model = PredictPriceFromModel("whatever.csv")
    price_no_model.ask_for_mileage()
    pout.as_title3('TEST2 : valid model')
    price_model = PredictPriceFromModel(theta_file)
    price_model.ask_for_mileage()


if __name__ == "__main__":
    test_predict_price_class("./gradient_descent_model/theta.csv")
