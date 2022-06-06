import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               FixedLocator, IndexLocator, LinearLocator)


def load_and_plot_tEMs():

    filename = '/Volumes/sel_external/ethiopia_irrigation/modis/4_tEMs_single_evergreen_double_dark.txt'

    cols = ['date', 'single', 'evergreen', 'double', 'dark' ]

    df = pd.read_fwf(filename, skiprows=range(6), names=cols)

    fig, ax = plt.subplots(figsize=(11,4.5))

    ticknames = ['01/2012', '01/2013', '01/2014', '01/2015', '01/2016', '01/2017',
                 '01/2018', '01/2019', '01/2020', '01/2021']
    # minors = np.linspace(0, len(df), len(df)+1)

    ax.plot(range(len(df)), df['single']/10000, color='red', label='Single Cycle')
    ax.plot(range(len(df)), df['evergreen']/10000, color='green', label='Evergreen')
    ax.plot(range(len(df)), df['double']/10000, color='blue', label='Double Cycle')
    ax.plot(range(len(df)), df['dark']/10000, color='k', label='Non-Vegetated')


    ax.set_ylabel('EVI', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_xticklabels(ticknames, rotation=40, fontsize=12)
    ax.xaxis.set_major_locator(IndexLocator(23, 13))
    ax.xaxis.set_minor_locator(IndexLocator(23, 2))
    plt.setp(ax.get_yticklabels(), fontsize=12)
    ax.legend(fontsize=12, loc='upper left')

    ax.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    load_and_plot_tEMs()