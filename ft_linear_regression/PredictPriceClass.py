import numpy as np
from my_colors import *

class PredictPriceFromModel():
    """ class for predicting price for a given mileage 
        based on linear regression parameters, theta = [θ0, θ1]
        self.__init__        : takes file name as parameter 
        self.__upload_model  : loads parameters from file
        self.ask_for_mileage : user enters a mileage
                             : will raise exceptions
        self.__predict_price : calculates the price based on theta
        self.__str__         : returns prediction result as a string
        
        Nested classes for Exceptions : out range mileages and prices
    """
    def __init__(self, file):
        """"""
        self.model_file = file
        self.theta = np.zeros(2)
        self.__upload_model()
        self.mileage = 0
        self.price = self.__predict_price(self.mileage)
        return None

    def __upload_model(self):
        try:
            self.theta = np.loadtxt(self.model_file)
            print("✅", f'{COL_BLUWHI}Linear regression model parameters loaded{COL_RESET}')
        except:
            print("❌", f'{COL_REDWHI}Linear regression model parameters not found : initialized to default{COL_RESET}')
        print(f'{COL_GRNBLK}\thypothesis :\n\tprice = θ0 + θ1 * mileage')
        print('\tθ0 = {:.4f}     θ1 = {:.4f}'.format(self.theta[0], self.theta[1]))
        print(f'{COL_RESET}')

    def __predict_price(self, mileage):
        """ 
        hypothesis : price = θ0 + θ1 * mileage 
        price is calculated with dot product :
        price = [θ0, θ1].[    1    ]
                         [ mileage ]
        """
        vec = np.ones(2)
        vec[1] = mileage
        price = np.dot(self.theta, vec)
        return price

    def ask_for_mileage(self):
        """ """
        try:
            in_str = input('\x1b[4;34;44m'
                    + 'Please, enter a car mileage (in km):'
                    + '\x1b[0m')
            self.mileage = float(in_str)
            if self.mileage < 0 or self.mileage > 1E6:
                raise InvalidMileageRangeError(self.mileage)
            self.price = self.__predict_price(self.mileage) 
            if self.price < 0:
                raise NegativePredictedPriceError()
        except (RecursionError, RuntimeError, TypeError, ValueError):
            print("Error : input is not valid")
        except (EOFError):
            print("Error : unexpected end of file")
        except (NegativePredictedPriceError):
            self.price = 0
            print("Mileage too High: predicted price is zero")
            print(self)
        except (InvalidMileageRangeError):
            print("Error: mileage is out of range")
        else:
            print(self)
        return None

    def __str__(self):
        return f'mileage = {format(self.mileage, ".0f")} km    predicted price = $ {format(self.price, ".2f")}'
    
class InvalidMileageRangeError(Exception):
    """ Exception raised for errors in the input Mileage"""
    def __init__(self, mileage, message="Invalid mileage : out of range"):
        self.mileage = mileage
        self.message = message
        super().__init__(self.message)

class NegativePredictedPriceError(Exception):
    """ Exception raised for errors in the input Mileage"""

    def __init__(self, message="Invalid price : out of range"):
        self.message = message
        super().__init__(self.message)

def test_predict_price_class(theta_file:str):
    """ tests for PredictPriceFromModel class """
    print(f'\n{COL_BLUWHI}----------- TEST MODE : PredictPriceFromModel class -----------{COL_RESET}\n')
    print(PredictPriceFromModel.__doc__)
    print(f'\n{COL_BLUCYA}----------- TEST1 : no model -----------{COL_RESET}\n')
    price_no_model = PredictPriceFromModel("./gradient_descent_model/whatever.csv")
    price_no_model.ask_for_mileage()
    print(f'\n{COL_BLUCYA}----------- TEST2 : valid model -----------{COL_RESET}\n')
    price_model = PredictPriceFromModel("./gradient_descent_model/theta.csv")
    price_model.ask_for_mileage()

if __name__ == "__main__":
    test_predict_price_class()
