import numpy as np

def upload_model_theta() -> np.ndarray:
    try:
        theta = np.loadtxt("./gradient_descent_model/theta.csv")
        print("linear regression model parameters loaded : [SUCCESS]") #dtype=int)
    except:
        theta = np.zeros(2)
        print("linear regression model parameters not found : initialized to default")
    return theta

def predicted_price(theta, mileage: int):
    """ price = t0 + t1 * mileage """
    vec = np.ones(2)
    vec[1] = mileage
    price = np.dot(theta, vec)
    return price

if __name__ == "__main__":
    theta = upload_model_theta()
    mileages = [0, 25000, 100000, 200000]
    for mileage in mileages:
        print('mileage = {} km    predicted price = {:.2f} $'.format(mileage, predicted_price(theta, mileage)))
