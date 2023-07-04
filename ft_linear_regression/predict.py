from training import LinearRegressionModel


def predict(model: LinearRegressionModel):
    """predicting a price form a given car mileage"""
    coeffs = model.get_coeffs()
    val = input('\x1b[6;30;43m' + 'Enter a car mileage:' + '\x1b[0m\n')
    print('\x1b[1;30;42m' + 'Estimated Price ($):' + '\x1b[0m')
    print('{:.2f}'.format(coeffs[0] + float(val) * coeffs[1]))
    return None


if __name__ == "__main__":
    """training model then predicting"""
    my_model = LinearRegressionModel()
    my_model.plot_data()
    predict(my_model)
