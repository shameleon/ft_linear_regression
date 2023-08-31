# : color code
RED = '\033[91m'
GRE = '\033[92m'
YEL = '\033[93m'
BLU = '\033[94m'
MAG = '\033[95m'
CYA = '\033[96m'
RES = '\033[0m'
BOL = '\033[1m'
UND = '\033[4m'

def normalize_array(arr):
    return ((arr - min(arr))) / (max(arr) - min(arr))

def denormalize_array(norm_array, arr):
    return (norm_array * (max(arr) - min(arr))) + min(arr)

def raw_estimated_price(t0, x, t1):
    return t0 + x * t1

def estimated_price(t0, t1, x, X, Y):
    price_ranged = raw_estimated_price(t0, (x - min(X)) / (max(X) - min(X)), t1)
    return price_ranged * (max(Y) - min(Y)) + min(Y)

def set_gradient_csv(output, t0, t1):
    # Theta file
    try:
        with open(output, "w+") as f:
            f.write('T0:{}\nT1:{}\n'.format(t0, t1))
    except:
        print('Wrong file')
        sys.exit(0)

    def RMSE_percent(self):
        self.RMSE = 100 * (1 - self.cost() ** 0.5)
        return self.RMSE

    def MSE_percent(self):
        self.MSE = 100 * (1 - self.cost())
        return self.MSE

    def cost(self):
        """
        MSE
        """
        dfX = DataFrame(self.X, columns=['X'])
        dfY = DataFrame(self.Y, columns=['Y'])
        return ((self.T1 * dfX['X'] + self.T0 - dfY['Y']) ** 2).sum() / self.M


def	storeData(t0, t1, file):
	with open(file, 'w') as csvfile:
		csvWriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		csvWriter.writerow([t0, t1])

from train import getPath, getData
from estimate import getThetas, estimatePrice

def getAccuracy(thetas, mileages, prices):
    price_average = sum(prices) / len(prices)
    ssr = sum(map(lambda mileage, price: pow(
        price - estimatePrice(thetas, mileage, mileages, prices), 2
    ), mileages, prices))
    sst = sum(map(lambda price: pow(price - price_average, 2), prices))
    return (1 - (ssr / sst))

def main():
    thetas = getThetas(getPath('thetas.csv'))
    mileages, prices = getData(getPath('data.csv'))
    accuracy = getAccuracy(thetas, mileages, prices)
    print("ft-linear-regression accuracy is: {}".format(accuracy))

if __name__ == "__main__":
    main()

def main() -> None:
    """ """
    print(f"{BLU}{BOL}{UND}[Loading]{RES}")
    pass


if __name__ == "__main__":
    """training model then predicting"""
    main()

# https://www.geeksforgeeks.org/gradient-descent-in-linear-regression/


x0, x1 = self.training_set[0][0], self.training_set[1][0]
x0n, x1n = self.normalized_training_set[0][0], self.normalized_training_set[1][0]
y0n, y1n = self.hypothesis(x0n), self.hypothesis(x1n)
p_diff = self.max_price - self.min_price

theta0 = (x1 / (x1 - x0)) * (y0n * p_diff + self.min_price - (x0 / x1 * (y1n * p_diff + self.min_price)))
y0 = self.training_set[0][1]

theta1 = (y0 - theta0) / x0
print(theta0, theta1) //RESULT: 8481.172796984529 -0.020129886654102203