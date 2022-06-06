import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.mask import mask
import numpy as np
import pandas as pd
import datetime as dt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               FixedLocator, IndexLocator, LinearLocator)
from dateutil.relativedelta import *
from dateutil.rrule import *
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from glob import glob

colors_xkcd = ['very dark purple', "windows blue", "amber", "greyish",
                   "faded green", 'pastel blue', 'grape', 'dark turquoise', 'terracotta',
                   'salmon pink', 'evergreen'
                   ]
cmap = sns.color_palette(sns.xkcd_palette(colors_xkcd), desat=1)
sns.set_palette(sns.xkcd_palette(colors_xkcd))
# sns.set_style("whitegrid")


def load_chirps_from_envi():
    df = pd.read_csv('additional_data/saved_from_envi/asasa_chirps_ts_2011-2021.txt', header=2, sep='s+',
                     names=['rainfall_mm_mo'])

    return list(df['rainfall_mm_mo'])


def load_evi_from_envi():
    df = pd.read_csv('additional_data/saved_from_envi/asasa_modis_ts_2011-2021.txt', header=2, sep='s+',
                     names=['evi'])

    return list(df['evi'])

def load_and_plot_chirps_and_modis_ts():

    chirps = '/Volumes/sel_external/ethiopia_irrigation/chirps/temporal_stacks/ethiopia_monthly_05152011_06152021_0.05_degrees.tif'
    shp = 'additional_data/shapefiles/amhara_rift_boundary_area.geojson'


    chirps_2011_2021 = load_chirps_from_envi()[1:-1]
    evi_2011_2021 = load_evi_from_envi() #[1:-1]

    evi_file_dir = '/Volumes/sel_external/ethiopia_irrigation/modis/individual_images/1000m'

    evi_tifs = glob(f'{evi_file_dir}/*.tif')
    evi_doys = [i.split('_')[-2].strip('doy') for i in evi_tifs]
    evi_dts = [dt.datetime.strptime(i, '%Y%j') for i in evi_doys]
    # print(evi_dts)

    # evi_dts = evi_dts[1:-1]

    print(len(chirps_2011_2021))
    print(len(evi_2011_2021))


    chirps_base = dt.date(year=2011, month=6, day=15)
    chirps_ts = list(rrule(freq=MONTHLY, count=120, dtstart=chirps_base, ))

    labels = [f'{i-1}-{i}' for i in range(2012, 2022)]


    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10,10))

    for ix in range(10):

        ln1 = axes[0].plot_date(chirps_ts[0:12], chirps_2011_2021[ix*12:(ix+1)*12], # color=cmap[0],
                              marker=None, linestyle='-', label=labels[ix])
        ln2 = axes[1].plot_date(evi_dts[0:23], evi_2011_2021[ix*23:(ix+1)*23], #color=cmap[1],
                              marker=None, linestyle='-', label=labels[ix])

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1,3,5,7,9,11)))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

        for label in ax.get_xticklabels(which='major'):
            label.set(rotation=45, horizontalalignment='center', fontsize=12)

    axes[0].set_ylabel('CHIRPS Monthly Rainfall [mm/mo]', color='k', fontsize=12)
    axes[1].set_ylabel('MODIS EVI', color='k', fontsize=12)
    axes[1].set_xlabel('Date', fontsize=12)
    axes[0].tick_params(axis='y', labelsize=12)
    axes[1].tick_params(axis='y', labelsize=12)

    axes[0].grid(True)
    axes[1].grid(True)

    axes[1].legend(loc='upper right', fontsize=12)
    # ax1.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':

    load_and_plot_chirps_and_modis_ts()
