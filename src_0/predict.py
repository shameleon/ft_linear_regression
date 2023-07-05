from training import LinearRegressionModel


def predict(model: LinearRegressionModel):
    """predicting a price form a given car mileage"""
    coeffs = model.get_coeffs()
    try:
        val = int(input('\x1b[6;30;43m'
                        + 'Please enter a car mileage:'
                        + '\x1b[0m\n'))
        print('\x1b[1;30;42m'
            + 'Estimated Price ($):'
            + '\x1b[0m')
        print('{:.2f}'.format(coeffs['origin'] + float(val) * coeffs['slope']))
    except ValueError:
        print("Oops!  That was no valid mileage.")
    return None


if __name__ == "__main__":
    """training model then predicting"""
    for cls in [LinearRegressionModel]:
        try:
            my_model = LinearRegressionModel()
            print(my_model)
            my_model.plot_data()
        except:
            print("Error : cannot build model from training")
    while True:
        try:
            predict(my_model)
        except (RuntimeError, TypeError, NameError):
            print("Error : cannot predict")

