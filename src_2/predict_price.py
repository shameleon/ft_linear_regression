import os.path
import pandas as pd

def main() -> None:
    """ """
    check_dir = os.path.exists("./model")
    print (check_dir)
    path = "./model/coeffs.csv"
    check_file = os.path.isfile(path)
    print (check_file)
    file = open(path, 'r')
    lines = file.readlines()
    # https://www.geeksforgeeks.org/read-a-file-line-by-line-in-python/
    # coeffs = pd.read_csv(path, sep=",", usecols=['theta0', 'theta1'])
    # data = pd.read_csv(path, sep=",", usecols=['theta0', 'theta1'])
    if not lines[1]:
        print("empty")
    else:
        print(lines[1]) #0,0


if __name__ == "__main__":
    """predicting price for a given mileage"""
    main()

# https://www.geeksforgeeks.org/gradient-descent-in-linear-regression/
# https://docs.python.org/3/tutorial/errors.html
