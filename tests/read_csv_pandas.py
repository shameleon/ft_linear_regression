import numpy as np
import pandas as pd
import sys
import os

def import_pd_dataset(file: str) -> pd.DataFrame:
    try:
        path_to_file = os.getcwd() + '/' + file
        df1 = pd.read_csv(f'{path_to_file}', sep=",", usecols=['km', 'price'])
        df2 = pd.read_csv(f'{file}')
    except:
        print("pandas read_csv : File not found.")
        sys.exit(0)
    return df2

def main() -> None:
    """ """
    df = import_pd_dataset('data.csv')
    print(df.head())
    labels = df.keys()
    arr = df.to_numpy()
    x_train = arr[:,0]
    y_train = arr[:,1]


if __name__ == "__main__":
    """ """
    main()
