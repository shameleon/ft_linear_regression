from train_model import LinearRegressionModel


class   PredictPrice:
    """ """
    def __init__(self, coeffs):
        self.coeffs = coeffs
        self.max_mileage = 500000.0
        print(self.coeffs['origin'])
        print(self.coeffs['slope'])
        #set_max_mileage()
        #self.loop()
        return None

    def set_max_mileage(self):
        """ price = 0 = t0 + t1 * mileage """
        if coeffs['slope'] != 0:
            self.max_mileage = - self.coeffs['origin'] / self.coeffs['slope']
        print(self.max_mileage)
        return None

    def predicted_price(self, mileage: int):
        """ price = t0 + t1 * mileage """
        price = self.coeffs['origin'] + self.coeffs['slope'] * mileage
        if (price < 0):
            price = 0.0
        print(price)
        return price

    def loop(self):
        """ """
        try:
            ask_for_mileage()
        except ValueError:
            print("Oops!  That was no valid mileage.")

    def ask_for_mileage(self):
        """ """
        in_str = input('\x1b[6;30;43m'
                    + 'Please enter a car mileage:'
                    + '\x1b[0m\n')
        if (in_str == "q"):
            raise ValueError
        val = int(stdin)
        if (val < 0 ) * (val > self.max_mileage):
            raise ValueError
        price = predicted_price(val)
        print('\x1b[1;30;42m'
            + 'Predicted Price ($):'
            + '\x1b[0m')
        print('{:.2f}'.format(price))
        return None
    
    def __str__(self):
        print ('Predicting price for a given car mileage, based on a trained model :')
        print ('\x1b[6;30;60m price = {1} + mileage * {2} \x1b[0m',
                self.coeffs['origin'],
                self.coeffs['slope'])
        return None


if __name__ == "__main__":
    """training model then predicting"""
    my_model = LinearRegressionModel()
    for i in range(3):
        try:
            predict_from_model = PredictPrice(my_model.get_coeffs())
            print(predict_from_model)
            break
        except (RuntimeError):
            print("Error : Runtime ")
        except (TypeError, NameError):
            print("Error : TypeError or NameError ")
        except ValueError:
            print("Oops!  That was no valid mileage.")

#https://www.geeksforgeeks.org/gradient-descent-in-linear-regression/
#https://docs.python.org/3/tutorial/errors.html