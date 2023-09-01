import numpy as np
import pandas as pd

def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(f'{file}')
    return df

def update_predicted_output(x_train):
    


def main():
    df = load_data('data.csv')
    labels = df.keys()
    arr = df.to_numpy()
    x_train = arr[:,0]
    y_train = arr[:,1]

if __name__ == "__main__":
    main()
