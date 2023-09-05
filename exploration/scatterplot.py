import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def sns_resid_plot(df):
    """ """
    sns.set_theme(style="darkgrid")
    sns.residplot(data=df, x="km", y="price")
    plt.show()

def sns_join_plot(df):
    """ """
    sns.set_theme(style="darkgrid")
    #plt.figure(figsize=(16,16))
    sns.jointplot(x="km", y="price", data=df,
                    kind="reg", truncate=False,
                    xlim=(0, 300000), ylim=(0, 10000),
                    color="m", height=7)
    plt.show()

def main():
    """importing csv then calling plots"""
    df = pd.read_csv('data.csv', sep=",", usecols=['km', 'price'])
    print('Imported data.csv')
    sns_join_plot(df)
    sns_resid_plot(df)


if __name__ == "__main__":
    main()

# https://seaborn.pydata.org/generated/seaborn.regplot.html
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html#sphx-glr-gallery-lines-bars-and-markers-scatter-hist-py