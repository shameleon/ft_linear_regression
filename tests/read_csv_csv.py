import numpy as np
import csv
import sys

def import_csv_dataset(path_to_file: str) -> np.array:
    try:
        with open(path_to_file, 'r') as f:
            reader = csv.reader(f)
            dataset = [line for line in reader]
            f.close()
            labels = dataset[0]
            arr = np.asarray(dataset[1:], dtype=np.float_)
            print(arr)
    except:
        print("csv.reader  :", path_to_file, "=======> File not found.")
        sys.exit(0)
    return arr

def main() -> None:
    """ """
    arr = import_csv_dataset('./datasets/data.csv')
    x_train = arr[:,0]
    # y_train = arr[:,1]
    print('x_train', x_train)
    print('x_mean', np.mean(x_train))
    # print('y_train', y__train)

if __name__ == "__main__":
    """ """
    main()
