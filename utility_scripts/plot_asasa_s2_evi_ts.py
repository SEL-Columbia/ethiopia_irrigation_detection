import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import datetime as dt

import seaborn as sns

colors_xkcd = ['very dark purple', "windows blue", "amber", "greyish",
                   "faded green", 'pastel blue', 'grape', 'dark turquoise', 'terracotta',
                   'salmon pink', 'evergreen'
                   ]
cmap = sns.color_palette(sns.xkcd_palette(colors_xkcd), desat=1)
sns.set_palette(sns.xkcd_palette(colors_xkcd))
# sns.set_style('whitegrid')

def plot_timeseries():

    main_dir = 'additional_data/dl_downloaded_ts/asasa'
    csvs = glob(f'{main_dir}/*.csv')

    fig, ax = plt.subplots()

    for csv in csvs:

        df = pd.read_csv(csv)

        evi_dates = [dt.datetime.strptime(i.split(' ')[0], '%Y-%m-%d') for i in list(df['vi_dates'])]
        evi_ts = list(df['evi'])

        ln = ax.plot_date(evi_dates, evi_ts, marker=None, linestyle='-', label='Sentinel-2 EVI')



    ax.set_ylabel('EVI', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.tick_params(axis='both', labelsize=12)

    ax.grid()
    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    plot_timeseries()