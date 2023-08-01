from train_model import LinearRegressionModel


class PredictPrice:
    """ """
    def __init__(self, coeffs):
        self.coeffs = coeffs
        self.max_mileage = 500000.0
        self.set_max_mileage()
        # self.loop()
        return None

    def set_max_mileage(self):
        if self.coeffs['slope'] != 0:
            self.max_mileage = - self.coeffs['origin'] / self.coeffs['slope']
        print("max mileage: ", self.max_mileage)
        return None

    def predicted_price(self, mileage: int):
        """ price = t0 + t1 * mileage """
        price = self.coeffs['origin'] + self.coeffs['slope'] * mileage
        if (price < 0):
            price = 0.0
        print(price)
        return price

    def ask_for_mileage(self):
        """ """
        in_str = input('\x1b[6;30;43m'
                       + 'Please enter a car mileage:'
                       + '\x1b[0m\n')
        if (in_str == "q"):
            raise ValueError
        try:
            val = int(in_str)
            if (val < 0) * (float(val) > self.max_mileage):
                raise ValueError
        except (RuntimeError, TypeError, ValueError):
            print("Error : Invalid Mileage")
        else:
            price = self.predicted_price(val)
            print('\x1b[1;30;42m' + 'Predicted Price ($):' + '\x1b[0m')
            print('{:.2f}'.format(price))
        return None

    def loop(self):
        """ """
        try:
            self.ask_for_mileage()
        except ValueError:
            print("Oops!  That was no valid mileage.")

    def __str__(self):
        a = self.coeffs['origin']
        b = self.coeffs['slope']
        return f'Predicting price for a given car mileage,\n\
                based on a trained model.\n\
                \x1b[6;30;60m    price = {a} + mileage * {b} \x1b[0m'


def main() -> None:
    """ """
    my_model = LinearRegressionModel()
    coeffs = my_model.get_coeffs()
    for i in range(3):
        try:
            predict_from_model = PredictPrice(coeffs)
            print(predict_from_model)
            predict_from_model.loop()
            break
        except (RuntimeError, TypeError):
            print("Error : Runtime ")
        except (NameError):
            print("Error : TypeError or NameError ")
        except ValueError:
            print("Oops!  That was no valid mileage.")


if __name__ == "__main__":
    """training model then predicting"""
    main()

# https://www.geeksforgeeks.org/gradient-descent-in-linear-regression/
# https://docs.python.org/3/tutorial/errors.html
