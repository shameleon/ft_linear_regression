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


class PredictPriceFromModel():
    """ [θ0, θ1] """
    def __init__(self, file):
        self.model = file
        self.theta = np.zeros(2)
        self.upload_model()

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
        in_str = input('\x1b[6;30;43m'
                    + 'Please enter a car mileage:'
                    + '\x1b[0m\n')
        
def intro():
    print(f'\n{COL_BLUWHI}----------- PREDICT A CAR PRICE -----------{COL_RESET}\n\n')
    
def ask_for_mileage(self):
        """ """

def main():
    intro()
    my_model = PredictPriceFromModel("./gradient_descent_model/theta.csv")
    mileages = [0, 25000, 100000, 200000, 35236, 89465.8, -5000, 4597866]
    for mileage in mileages:
        pred_price = my_model.predict_price(mileage)
        print('mileage = {} km    predicted price = {:.2f} $'.format(mileage, pred_price))

if __name__ == "__main__":
    main()