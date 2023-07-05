import pandas as pd
import matplotlib.pyplot as plt


def subplots(df1, df2):
    """ """
    #fig = plt.figure()
    #ax1 = fig.add_subplot(2, 3, 1)
    #plt.subplot(211)
    plt.scatter(df1["km"], df1["price"], c='orange')
    plt.pause(0.5)
    plt.scatter(df2["km"], df2["price"], c='red')
    plt.show()

def main():
    """importing csv then calling plots"""
    url = 'https://cdn.intra.42.fr/document/document/11434/data.csv'
    df1 = pd.read_csv(url, sep=",", usecols=['km', 'price'])
    print('Imported data.csv')
    data2 = {'km': [750, 0, 5000, 2000, 6000, 0],
         'price': [12000, 13500, 8900, 9000, 8000, 15000]
         }  
    df2 = pd.DataFrame(data2)
    subplots(df1, df2)


if __name__ == "__main__":
    main()

# https://seaborn.pydata.org/generated/seaborn.regplot.html
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html#sphx-glr-gallery-lines-bars-and-markers-scatter-hist-py