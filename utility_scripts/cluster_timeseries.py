import numpy as np
import geopandas as gpd
import rasterio
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from matplotlib import colors
from matplotlib.ticker import FixedLocator, IndexLocator
import descarteslabs as dl
import datetime
from dateutil.rrule import rrule, MONTHLY


colors_xkcd = ['very dark purple', "windows blue", "amber", "greyish",
                   "faded green", 'pastel blue', 'grape', 'dark turquoise', 'terracotta',
                   'salmon pink', 'evergreen'
                   ]
cmap = sns.color_palette(sns.xkcd_palette(colors_xkcd), desat=1)
sns.set_palette(sns.xkcd_palette(colors_xkcd))
sns.set_style("whitegrid")


def load_and_save_chirps_timeseries(region):

    dir = '/Volumes/sel_external/ethiopia_survey/survey_data/processed/bboxs'
    poly_file = f'{dir}/{region}_labels_bbox.geojson'

    geom = gpd.read_file(poly_file)['geometry'].iloc[0]

    aoi = dl.scenes.AOI(geom, resolution=0.05)
    start_date = '2017-01-01'
    end_date = '2019-12-31'


    rainfall_scenes, ctx = dl.scenes.search(
        aoi,
        products='9a638ef860cf9d231775813e2b65241da41f576f:chirps_monthly_precipitation_tc',
        start_datetime=start_date,
        end_datetime=end_date,
        limit=None,
    )

    vi_scenes = rainfall_scenes.groupby(
        'properties.date.year',
        'properties.date.month',
    )

    dt_strt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    dt_end = datetime.datetime.strptime(end_date, '%Y-%m-%d')

    date_tuples = [(dt.year, dt.month) for dt in rrule(MONTHLY,
                                                       dtstart=dt_strt,
                                                       until=dt_end)]

    rainfall_ts = np.zeros(len(date_tuples))

    for ix, (dt_tuple, month_scenes) in enumerate(vi_scenes):
        (year, month) = dt_tuple

        month_index = np.argwhere([i == dt_tuple for i in date_tuples])[0][0]
        # Collect rainfall scenes

        rainfall_img = month_scenes.mosaic(bands='monthly_precipitation',
                                       ctx=ctx,
                                       bands_axis=-1)

        rainfall_avg = np.mean(rainfall_img)



        rainfall_ts[month_index] = rainfall_avg

    save_dir = f'/Volumes/sel_external/ethiopia_survey/saved_imagery/{region}/chirps_rainfall'
    file_ext_evi = f'{save_dir}/{region}_monthly_rainfall_avg_mm.csv'
    pd.DataFrame(data=rainfall_ts).to_csv(file_ext_evi)


def load_pixel_timeseries(region, layer_type, irrig=True):
    print(f'Loading pixel timeseries for region {region}, irrigation = {irrig}')

    if irrig:
        irrig_str = 'irrig'
    else:
        irrig_str = 'noirrig'

    dir = f'/Volumes/sel_external/ethiopia_survey/saved_imagery/{region}/labeled_pixels'
    ts_file = f'{dir}/{region}_{irrig_str}_labeled_pixels_{layer_type}_monthly.csv'

    ts_df = pd.read_csv(ts_file, index_col=0)

    print(len(ts_df))
    return ts_df



def pca_and_cluster_timeseries(region):

    layer_type = 'ndwi'

    n_pca_components = 7
    n_kmeans_clusters = 5

    irrig_ts_df = load_pixel_timeseries(region, layer_type, irrig=True)
    noirrig_ts_df = load_pixel_timeseries(region, layer_type, irrig=False)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    fig1, axes1 = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    ax12 = axes1.twinx()

    df_strings = ['irrigated', 'non-irrigated']

    parent_dir = f'/Volumes/sel_external/ethiopia_survey/saved_imagery/{region}'
    rainfall_ts = pd.read_csv(f'{parent_dir}/chirps_rainfall/{region}_monthly_rainfall_avg_mm.csv', index_col=0)

    ymins = []
    ymaxs = []

    for ix, df in enumerate([irrig_ts_df, noirrig_ts_df]):

        print(f'Calculate PCA, {df_strings[ix]}')
        pca = PCA(n_components=n_pca_components)

        principalComponents = pca.fit_transform(df)

        print(f'Cluster, {df_strings[ix]}')
        kmeans_cluster = KMeans(n_clusters=n_kmeans_clusters).fit(principalComponents)
        cluster_predicts = kmeans_cluster.predict(principalComponents)
        cluster_centers = kmeans_cluster.cluster_centers_

        cluster_centers_ts = pca.inverse_transform(cluster_centers)
        # cluster_centers_ts = np.clip(cluster_centers_ts, a_min=32767.5, a_max=None)


        print(f'Plotting, {df_strings[ix]}')
        ax2 = axes[ix].twinx()

        for i in range(n_kmeans_clusters):
            axes[ix].plot(range(cluster_centers_ts.shape[1]), (cluster_centers_ts[i] - 32767.5)/32767.5,
                         label=f'Cluster {i}, {np.count_nonzero(cluster_predicts == i)} px.',
                         color=cmap[i+1])


            if ix == 0:
                plot_color = cmap[1]
            else:
                plot_color = cmap[2]

            if i == 0:
                axes1.plot(range(cluster_centers_ts.shape[1]), (cluster_centers_ts[i] - 32767.5)/32767.5,
                         color=plot_color, label = df_strings[ix])
            else:
                axes1.plot(range(cluster_centers_ts.shape[1]), (cluster_centers_ts[i] - 32767.5) / 32767.5,
                           color=plot_color)



        ticknames = ['01/2017', '07/2017', '01/2018', '07/2018', '01/2019', '07/2019',]
        minors = np.linspace(0, cluster_centers_ts.shape[1], 37)
        axes[ix].set_xlabel('Month')
        axes[ix].set_xticklabels(ticknames)
        axes[ix].xaxis.set_major_locator(IndexLocator(cluster_centers_ts.shape[1] / 6, 0))

        axes[ix].xaxis.set_minor_locator(FixedLocator(minors))
        axes[ix].tick_params(axis='x', which='minor', length=2)
        axes[ix].tick_params(axis='x', which='major', length=4)

        ax2.plot(range(36), rainfall_ts, label='CHIRPS')
        ax2.grid(False)

        axes[ix].legend(loc='lower right')
        ax2.legend(loc='upper right')

        axes[ix].set_ylabel(layer_type)
        ax2.set_ylabel('Rainfall [mm]')

        axes[ix].plot(range(cluster_centers_ts.shape[1]), np.tile(.3, cluster_centers_ts.shape[1]),
                     color=cmap[-1], linestyle=':')

        axes[ix].set_title(f'Clustered {df_strings[ix]} pixels, {region}')

        ymins.append(axes[ix].get_ylim()[0])
        ymaxs.append(axes[ix].get_ylim()[1])

    # Plot combined
    axes1.set_xlabel('Month')
    axes1.set_xticklabels(ticknames)
    axes1.xaxis.set_major_locator(IndexLocator(cluster_centers_ts.shape[1] / 6, 0))

    axes1.xaxis.set_minor_locator(FixedLocator(minors))
    axes1.tick_params(axis='x', which='minor', length=2)
    axes1.tick_params(axis='x', which='major', length=4)

    ax12.plot(range(36), rainfall_ts, label='CHIRPS')
    ax12.grid(False)
    axes1.set_ylabel(layer_type)
    axes1.set_title(f'Combined Irrig. and Non-Irrig. Clusters, {region}')
    ax12.set_ylabel('Rainfall [mm]')
    ax12.legend(loc='upper right')
    axes1.legend(loc='upper left')


    for ix, ax in enumerate(axes):
        ax.set_ylim([np.min(ymins), np.max(ymaxs)])

    fig.tight_layout()
    fig1.tight_layout()
    plt.show()

if __name__ == '__main__':
    region = 'tana'
    pca_and_cluster_timeseries(region)

