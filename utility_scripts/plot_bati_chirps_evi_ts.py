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
    df = pd.read_csv('additional_data/saved_from_envi/chirps_ts_envi.txt', header=2, sep='s+',
                     names=['rainfall_mm_mo'])

    return list(df['rainfall_mm_mo'])


def load_evi_from_envi():
    df = pd.read_csv('additional_data/saved_from_envi/modis_ts_envi_4.txt', header=2, sep='s+',
                     names=['evi'])

    return list(df['evi'])

def load_and_plot_chirps_and_s2_ts():

    chirps = '/Volumes/sel_external/ethiopia_irrigation/chirps/temporal_stacks/ethiopia_monthly_05152011_06152021_0.05_degrees.tif'
    shp = 'additional_data/shapefiles/amhara_rift_boundary_area.geojson'

    # poly = list(gpd.read_file(shp)['geometry'])
    #
    # with rasterio.open(chirps, 'r') as src:
    #     img, trans = mask(src, poly, crop=True)
    #
    #
    # chirps_2020_2021 = np.nanmean(img[-25::], axis=(1,2))
    # print(chirps_2020_2021)

    chirps_2020_2021 = load_chirps_from_envi()[-25::]

    ticknames = ['06/2019', '09/2019', '12/2019', '03/2020', '06/2020', '09/2020',
                 '12/2020', '03/2020', '06/2020', '09/2020', '12/2020', '03/2021',
                 '06/2021']

    chirps_base = dt.date(year=2019, month=6, day=15)
    chirps_ts = list(rrule(freq=MONTHLY, count=25, dtstart=chirps_base, ))


    ## Load ts
    df = pd.read_csv('additional_data/dl_downloaded_ts/amhara_rift_count_0.csv')
    evi_dates = [dt.datetime.strptime(i.split(' ')[0], '%Y-%m-%d') for i in list(df['vi_dates'])]
    evi_ts =  list(df['evi'])
    # print(dates)




    fig, ax = plt.subplots(figsize=(10,7))
    ax1 = ax.twinx()
    ln1 = ax.plot_date(chirps_ts, chirps_2020_2021, color=cmap[0], marker=None, linestyle='-', label='CHIRPS Rainfall')
    ln2 = ax1.plot_date(evi_dates, evi_ts, color=cmap[1], marker=None, linestyle='-', label='Sentinel-2 EVI')

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(3,6,9,12)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=45, horizontalalignment='center', fontsize=12)

    ax.set_ylabel('Monthly Rainfall [mm]', color=cmap[0], fontsize=12)
    ax1.set_ylabel('EVI', color=cmap[1], fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)

    ax.grid(True)

    ax.legend(loc='upper left', fontsize=12)
    ax1.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_chirps_by_year():

    chirps = load_chirps_from_envi()[1:-1]


    chirps_base = dt.date(year=2020, month=6, day=15)
    chirps_dates_ts = list(rrule(freq=MONTHLY, count=12, dtstart=chirps_base, ))


    chirps_yearly = np.reshape(chirps, (10,12))

    chirps_labels = [f'{i-1}-{i}' for i in range(2012, 2022)]

    fig, ax = plt.subplots(figsize=(10,7))
    for ix in range(chirps_yearly.shape[0]):
        ax.plot(chirps_dates_ts, chirps_yearly[ix], label=chirps_labels[ix])

    ax.legend(fontsize=12)
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.grid()
    ax.set_ylabel('Rainfall [mm/mo]', fontsize=12)
    ax.tick_params(axis='y', labelsize=12)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='center', fontsize=12)
        # label.replace(' 0', '')

    plt.tight_layout()
    plt.show()


def plot_modis_by_year():
    evi = load_evi_from_envi()
    evi_file_dir = '/Volumes/sel_external/ethiopia_irrigation/modis/individual_images/1000m'

    evi_tifs = glob(f'{evi_file_dir}/*.tif')
    evi_doys = [i.split('_')[-2].strip('doy') for i in evi_tifs]
    evi_dts = [dt.datetime.strptime(i, '%Y%j') for i in evi_doys]
    print(evi_dts)

    evi_dts_single_year = evi_dts[0:23]
    evi_yearly = np.reshape(evi, (10,23))

    evi_labels = [f'{i - 1}-{i}' for i in range(2012, 2022)]

    fig, ax = plt.subplots(figsize=(10, 7))
    for ix in range(evi_yearly.shape[0]):
        ax.plot(evi_dts_single_year, evi_yearly[ix], label=evi_labels[ix])

    ax.legend(fontsize=12)
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.grid()
    ax.set_ylabel('MODIS EVI', fontsize=12)
    ax.tick_params(axis='y', labelsize=12)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='center', fontsize=12)
        # label.replace(' 0', '')

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    # load_chirps_from_envi()
    load_and_plot_chirps_and_s2_ts()
