import numpy as np

# : color code
RED = '\033[91m'
GRE = '\033[92m'
YEL = '\033[93m'
BLU = '\033[94m'
MAG = '\033[95m'
CYA = '\033[96m'
COL_RESET = '\x1b[0m'
COL_GRNBLK = '\x1b[1;32;40m'
COL_BLUWHI = '\x1b[1;34;47m'
COL_BLURED = '\x1b[2;34;41m'
COL_REDWHI = '\x1b[2;31;47m'
COL_ERR = '\x1b[2;34;41m'


class PredictPriceFromModel():
    """ [θ0, θ1] """
    def __init__(self, file):
        self.model = file
        self.theta = np.zeros(2)
        self.upload_model()
        self.mileage = 0
        self.price = self.predict_price(self.mileage)

    def upload_model(self):
        try:
            self.theta = np.loadtxt(self.model)
            print("✅", f'{COL_BLUWHI}Linear regression model parameters loaded{COL_RESET}')
        except:
            print("❌", f'{COL_REDWHI}Linear regression model parameters not found : initialized to default{COL_RESET}')
        print(f'{COL_GRNBLK}\thypothesis :\n\tprice = θ0 + θ1 * mileage')
        print('\tθ0 = {:.4f}     θ1 = {:.4f}'.format(self.theta[0], self.theta[1]))
        print(f'{COL_RESET}')

    def predict_price(self, mileage):
        """ 
        hypothesis : price = θ0 + θ1 * mileage 
        dot product :
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
                    + 'Please enter a car mileage (in km):'
                    + '\x1b[0m')
            self.mileage = float(in_str)
            if self.mileage < 0 or self.mileage > 1E6:
                raise InvalidMileageRangeError(self.mileage)
            self.price = self.predict_price(self.mileage) 
            if self.price < 0:
                raise NegativePredictedPriceError()
        except (RecursionError, RuntimeError, TypeError, ValueError):
            print("Error : input is not valid")
        except (EOFError):
            print("Error : EOF is not a valid mileage, lol")
        except (NegativePredictedPriceError):
            self.price = 0
            print("Mileage too High: predicted price is zero")
            print(self)
        except (InvalidMileageRangeError):
            print("Error: mileage is out of range")
        else:
            print(self)

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


def intro():
    print(f'\n{COL_BLUWHI}----------- PREDICT A CAR PRICE -----------{COL_RESET}\n\n')

def test_predict_class(model:PredictPriceFromModel):
    mileages = [0, 25000, 100000, 200000, 35236, 89465.8, -5000, 4597866]
    for mileage in mileages:
        pred_price = model.predict_price(mileage)
        print('mileage = {} km    predicted price = {:.2f} $'.format(mileage, pred_price))


def main():
    intro()
    model_prediction = PredictPriceFromModel("./gradient_descent_model/theta.csv")
    # test_predict_class(model_prediction)
    continue_loop = True
    while(continue_loop):
        model_prediction.ask_for_mileage()
        # print(model_prediction)
        try:
            in_str = input('\r\x1b[2;37;40m\nContinue ? press [enter]\n [no] or [q] to quit\n \x1b[0m\r')
        except (EOFError):
            print("Error : EOF is not a option, lol")
        else:
            if (in_str in ["N", "No", "n", "no", "q", "Q", "exit", "quit", "QUIT"]):
                continue_loop = False


if __name__ == "__main__":
    main()

    """
    add line result to file """