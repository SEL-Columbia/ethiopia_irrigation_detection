import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import joblib
import sklearn
from matplotlib.ticker import FixedLocator, IndexLocator


import seaborn as sns

colors_xkcd = [ 'very dark purple', "windows blue", "amber", "greyish",
                   "faded green", 'pastel blue', 'terracotta', 'grape', 'dark turquoise',
                   'salmon pink', 'evergreen', 'royal blue', 'dark red'
                   ]


cmap = sns.color_palette(sns.xkcd_palette(colors_xkcd), desat=1)
cmap_base = sns.color_palette(sns.xkcd_palette(colors_xkcd), desat=0.4)
sns.set_palette(sns.xkcd_palette(colors_xkcd))

def load_saved_models(all_bands=True):
    base_dir = '../data/from_gcp/trained_models/random_forest'

    if all_bands:
        files = glob(f'{base_dir}/all_bands/*.joblib')
    else:
        files = glob(f'{base_dir}/evi_only_shifted/*.joblib')


    return files


def load_evi_files(regions):

    mean_evis = np.zeros((0, 36))

    for region in tqdm(regions):
        fn = f'/Volumes/sel_external/ethiopia_irrigation/{region}/training_data/csvs/evi_only_shifted/' \
             f'{region}_training_evi_only_labeled_cleaned_pixels.csv'

        df = np.array(pd.read_csv(fn, index_col=0).values)
        data = df[:, 0:36]
        labels = df[:, -1]
        irrig_ix = np.where(labels==2)

        irrig_mean = np.mean(data[irrig_ix], axis=0)[None,...]

        mean_evis = np.concatenate((mean_evis, irrig_mean))


    fig, ax = plt.subplots(figsize=(9, 7))
    for ix in range(len(regions)):
        ax.plot(range(36), mean_evis[ix], label=regions[ix])

    ax.legend()
    ax.grid(True)

    ticknames = ['06/2020', '08/2020', '10/2020', '12/2020', '02/2021', '04/2021', ]

    ## Set X ticks
    minors = np.linspace(0, 36, 37)
    ax.set_xlabel('Timestep')
    ax.set_xticklabels(ticknames, rotation=30)
    ax.xaxis.set_major_locator(IndexLocator(6, 0))

    ax.xaxis.set_minor_locator(FixedLocator(minors))
    ax.tick_params(axis='x', which='minor', length=2)
    ax.tick_params(axis='x', which='major', length=4)

    ax.set_ylabel('EVI')
    # ax.set_title('Mean EVI of Irrigated Samples by Region')

    plt.tight_layout()
    plt.show()



    return mean_evis






def plot_all_bands_results():

    all_bands_files = sorted(load_saved_models(all_bands=True))
    evi_only_files = load_saved_models(all_bands=False)

    all_bands_results = np.zeros((7, 36, 11))
    all_bands_results_count = np.zeros((7))

    evi_only_results = np.zeros((7, 36, 1))
    evi_only_results_count = np.zeros((7))



    for file in tqdm(all_bands_files):
        rf = joblib.load(file)
        try:
            feature_importances = rf.feature_importances_
            feature_importances = np.reshape(feature_importances, (36, 11))

            in_regions = len(file.split('/')[-1].split('_')[-1].strip('.csv').split('-')) -1
            all_bands_results[in_regions] += feature_importances
            all_bands_results_count[in_regions] += 1

        except Exception as e:
            print(e)
            print(file)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,7))
    for ix in range(0,7):
        axes[0].plot(range(36), np.mean(all_bands_results[ix], axis=-1)/all_bands_results_count[ix],
                     label=f'{ix+1} Regions, n={int(all_bands_results_count[ix])}')
        axes[1].plot(range(11), np.mean(all_bands_results[ix], axis=0)/all_bands_results_count[ix],
                     label=f'{ix+1} Regions, n={int(all_bands_results_count[ix])}')


    for ax in axes:
        ax.legend()
        ax.grid(True)

    ticknames = ['06/2020', '08/2020', '10/2020', '12/2020', '02/2021', '04/2021', ]
    band_names = ['Blue', 'Green', 'Red', 'RE-1', 'RE-2', 'RE-3', 'NIR', 'RE-4', 'SWIR1',
                  'SWIR2', 'CHIRPS']

    ## Set X ticks
    minors = np.linspace(0, 36, 37)
    axes[0].set_xlabel('Timestep')
    axes[0].set_xticklabels(ticknames, rotation=30)
    axes[0].xaxis.set_major_locator(IndexLocator(6, 0))

    axes[0].xaxis.set_minor_locator(FixedLocator(minors))
    axes[0].tick_params(axis='x', which='minor', length=2)
    axes[0].tick_params(axis='x', which='major', length=4)

    axes[1].set_xticks(range(11))
    axes[1].set_xticklabels(band_names, rotation=30)
    axes[1].set_xlabel('Band')


    axes[0].set_ylabel('Random Forest Normalized Mean Timestep Importance')
    axes[1].set_ylabel('Random Forest Normalized Mean Band Importance')

    plt.tight_layout()
    plt.show()

def plot_evi_only_single_regions():

    evi_only_files = load_saved_models(all_bands=False)

    evi_only_files = [file for file in evi_only_files if (
                      (len(file.split('/')[-1].split('_')[-1].strip('.csv').split('-')) == 1)
                    )]
                    #   and
                    # 'kobo' not in file.split('/')[-1].split('_')[-1].strip('.csv').split('-') and
                    # 'alamata' not in file.split('/')[-1].split('_')[-1].strip('.csv').split('-'))]


    feature_importances_stacked_ = np.zeros((0, 36))

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 7))

    for ix, file in tqdm(enumerate(evi_only_files)):
        print(file)

        rf = joblib.load(file)
        region = file.split('_')[-1].replace('.joblib', '')

        try:
            feature_importances = rf.feature_importances_
            feature_importances = np.reshape(feature_importances, (1, 36))

            feature_importances_stacked_ = np.concatenate((feature_importances_stacked_, feature_importances), axis=0)

            if ix >= 20:
                linestyle = '-.'
            elif ix >= 10:
                linestyle = ':'
            else:
                linestyle = '-'

            axes.plot(range(36), feature_importances[0], linestyle=linestyle, label=f'{region.capitalize()}')

        except Exception as e:
            print(e)


    # fig, ax = plt.subplots()
    # ax.plot(range(36), np.mean(feature_importances_stacked_, axis=0))
    # plt.show()

    axes.legend()
    axes.grid(True)

    ticknames = ['06/2020', '08/2020', '10/2020', '12/2020', '02/2021', '04/2021', ]

    ## Set X ticks
    minors = np.linspace(0, 36, 37)
    axes.set_xlabel('Timestep')
    axes.set_xticklabels(ticknames, rotation=30)
    axes.xaxis.set_major_locator(IndexLocator(6, 0))

    axes.xaxis.set_minor_locator(FixedLocator(minors))
    axes.tick_params(axis='x', which='minor', length=2)
    axes.tick_params(axis='x', which='major', length=4)

    axes.set_ylabel('Random Forest Normalized Mean Timestep Importance')

    plt.tight_layout()
    plt.show()

def plot_evi_only_results():


    evi_only_files = load_saved_models(all_bands=False)

    evi_only_results = np.zeros((7, 36, 1))
    evi_only_results_count = np.zeros((7))

    for file in tqdm(evi_only_files):
        rf = joblib.load(file)
        try:
            feature_importances = rf.feature_importances_
            feature_importances = np.reshape(feature_importances, (36, 1))

            in_regions = len(file.split('/')[-1].split('_')[-1].strip('.csv').split('-')) - 1
            evi_only_results[in_regions] += feature_importances
            evi_only_results_count[in_regions] += 1

        except Exception as e:
            print(e)
            print(file)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 7))
    for ix in range(0, 7):
        axes.plot(range(36), np.mean(evi_only_results[ix], axis=-1) / evi_only_results_count[ix],
                  label=f'{ix + 1} Regions, n={int(evi_only_results_count[ix])}')

    axes.legend()
    axes.grid(True)

    ticknames = ['06/2020', '08/2020', '10/2020', '12/2020', '02/2021', '04/2021', ]

    ## Set X ticks
    minors = np.linspace(0, 36, 37)
    axes.set_xlabel('Timestep')
    axes.set_xticklabels(ticknames, rotation=30)
    axes.xaxis.set_major_locator(IndexLocator(6, 0))

    axes.xaxis.set_minor_locator(FixedLocator(minors))
    axes.tick_params(axis='x', which='minor', length=2)
    axes.tick_params(axis='x', which='major', length=4)

    axes.set_ylabel('Random Forest Normalized Mean Timestep Importance')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    plot_evi_only_single_regions()

