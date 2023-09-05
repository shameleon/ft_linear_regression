import numpy as np


class LinearRegressionModel:

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.theta = np.zeros(2)

    def set_theta(self, bias:float, weight:float):
        self.theta[0] = bias
        self.theta[1] = weight

    def predict_price(self, mileage:float) -> float:
        """ 
        hypothesis : price = θ0 + θ1 * mileage 
        price is calculated with dot product :
        estimated price = [θ0, θ1].[    1    ]
                                   [ mileage ]
        """
        vec = np.ones(2)
        vec[1] = mileage
        price = np.dot(self.theta, vec)
        return price

    def predict_price1(self) -> np.ndarray:
        y_pred = np.empty(len(self.x_train))
        i = 0
        for i in range(len(self.x_train)):
            vec = np.ones(2)
            vec[1] = self.x_train[i]
            y_pred[i] = np.dot(self.theta, vec)
            i += 1
        return y_pred

    def predict_price2(self) -> np.ndarray:
        ones = np.ones(len(self.x_train))
        y_pred = np.array([])
        for left 
        np.dot(self.theta, np.array(1, x), for x in self.x_train)
        for i in range(len(self.x_train)):
            vec = np.ones(2)
            vec[1] = self.x_train[i]
            np.dot(self.theta, vec)
        print(self.theta)
        #arr = np.insert
        #y_pred = np.dot(self.theta, ) for i in range(len(self.x_train))


    def predict_price_for_mileage(self, mileage:float) -> float:
        """ 
        hypothesis : price = θ0 + θ1 * mileage 
        price is calculated with dot product :
        estimated price = [θ0, θ1].[    1    ]
                                   [ mileage ]
        """
        vec = np.ones(2)
        vec[1] = mileage
        price = np.dot(self.theta, vec)
        return price


def test_LinearRegressionModel_class() -> None:
    x_train = np.array([0.1, 0.3, 0.4, 0.8, 1])
    y_train = np.array([ 4, 2.5, 1.5, -1, -1.5  ])
    my_model = LinearRegressionModel(x_train, y_train)
    my_model.set_theta(4.335227, -6.221677)
    print(my_model.predict_price2())
    return None


if __name__ == "__main__":
    test_LinearRegressionModel_class()