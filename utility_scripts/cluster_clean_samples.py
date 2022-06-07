import numpy as np
from tqdm import tqdm
import sys, os
from glob import glob
import pandas as pd



from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from imagery_evaluation.gmm_explorations import extract_ndvi_clipped, extract_ndvi
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               FixedLocator, IndexLocator, LinearLocator)
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


colors_xkcd = [ 'very dark purple', "windows blue", "amber", "greyish",
                   "faded green", 'pastel blue', 'terracotta', 'grape', 'dark turquoise',
                   'salmon pink', ]# 'evergreen', 'royal blue', 'dark red'



cmap = sns.color_palette(sns.xkcd_palette(colors_xkcd), desat=1)
cmap_base = sns.color_palette(sns.xkcd_palette(colors_xkcd), desat=0.4)
sns.set_palette(sns.xkcd_palette(colors_xkcd))



def extract_evi(data):

    data = np.reshape(data, (data.shape[0], 36, 13)).astype(np.float32)
    evi = 2.5*(data[..., 7] - data[..., 3]) / (data[..., 7] + 6*data[..., 3] - 7.5*data[..., 1] + 10000)

    print(f'EVI shape: {evi.shape}')


    return evi

def clean_csvs(region):
    dir_name = f'/Volumes/sel_external/ethiopia_irrigation/{region}/training_data/csvs/full'

    configs = ['training', 'validation', 'testing']

    full_df_list = []
    nsamples_dict = {}

    for config in configs:
        nsamples_dict[f'{config}_total_samples'] = 0

        csvs = glob(f'{dir_name}/*_{config}_*.csv')

        config_only_df_list = []

        for csv in csvs:
            df = pd.read_csv(csv, index_col=0)
            full_df_list.append(df)
            config_only_df_list.append(df)

            nsamples_dict[f'{config}_total_samples']+=len(df)

        ## Determine # samples in each classs for each config
        config_only_data = np.array(pd.concat(config_only_df_list))
        config_only_noirrig = config_only_data[np.where(config_only_data[:, -1] == 1)]
        config_only_irrig = config_only_data[np.where(config_only_data[:, -1] == 2)]

        nsamples_dict[f'{config}_samples_per_class'] = [len(config_only_noirrig), len(config_only_irrig)]


    for key, values in nsamples_dict.items():
        print(f'{key}:{values}')

    full_data = np.array(pd.concat(full_df_list))


    noirrig_data_full = full_data[np.where(full_data[:, -1] == 1)]
    irrig_data_full = full_data[np.where(full_data[:, -1] == 2)]

    n_samples = 100000

    noirrig_data = noirrig_data_full[:, 5*13:41*13]
    irrig_data = irrig_data_full[:, 5*13:41*13]


    # data_list = [extract_ndvi_clipped(noirrig_data), extract_ndvi_clipped(irrig_data)]
    data_list = [extract_evi(noirrig_data), extract_evi(irrig_data)]

    # print(np.count_nonzero(np.isnan(data_list[0])))
    # print(np.count_nonzero(np.isnan(data_list[1])))
    # print(np.count_nonzero(np.isinf(data_list[0])))
    # print(np.count_nonzero(np.isinf(data_list[1])))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18,8))
    n_cluster_list = [15, 15]
    n_pca_components = 10
    pred_names = ['Non-irrigated', 'Irrigated']
    ticknames = ['Jun 2020', 'Aug 2020', 'Oct 2020', 'Dec 2020', 'Feb 2021', 'Apr 2021']
    minors = np.linspace(0, 36, 37)

    ylabels_list = []

    for ix, data in enumerate(data_list):
        ax = axes[ix]

        n_clusters = n_cluster_list[ix]

        pca = PCA(n_components=n_pca_components)
        principalComponents = pca.fit_transform(data)

        X = principalComponents
        gmm = GaussianMixture(n_components=n_clusters,
                              covariance_type='full')
        ylabels = gmm.fit_predict(X)
        ylabels_list.append(ylabels)

        for jx in range(n_clusters):
            valid_indices = np.where(ylabels == jx)[0]
            cluster_data = data[valid_indices]

            # ndvi = extract_ndvi(cluster_data)
            ndvi = cluster_data
            ndvi_mean = np.clip(np.mean(ndvi, axis=0), 0, 1)

            if jx >= 10:
                linestyle = '--'
            else:
                linestyle = '-'

            ax.plot(range(len(ndvi_mean)), ndvi_mean, linestyle=linestyle,
                    label = f'Cluster {jx}, {np.count_nonzero(ylabels==jx)} px.')
        fs = 15

        if ix == 0:
            ax.legend(loc='upper right', fontsize=fs-2)
        else:
            ax.legend(loc='upper left', fontsize=fs-2)

        ax.set_title(f'Total Pixels: {len(data)}', fontsize=fs)
        ax.set_ylabel(f'EVI', fontsize=fs)
        ax.set_xticklabels(ticknames, rotation=40, fontsize=fs)
        ax.xaxis.set_major_locator(IndexLocator(6, 0))
        ax.xaxis.set_minor_locator(FixedLocator(minors))
        ax.tick_params(axis='both', which='major', length=5, labelsize=fs)
        ax.tick_params(axis='x', which='minor', length=3)
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    noirrig_data_full = np.concatenate((noirrig_data_full, ylabels_list[0][..., None]), axis=-1)
    irrig_data_full = np.concatenate((irrig_data_full, ylabels_list[1][..., None]), axis=-1)

    data_dict = {}

    noirrig_ixs = [nsamples_dict[f'training_samples_per_class'][0],
                            (nsamples_dict[f'training_samples_per_class'][0] +
                             nsamples_dict[f'validation_samples_per_class'][0]),
                            (nsamples_dict[f'training_samples_per_class'][0] +
                             nsamples_dict[f'validation_samples_per_class'][0] +
                             nsamples_dict[f'testing_samples_per_class'][0])
                            ]

    irrig_ixs = [nsamples_dict[f'training_samples_per_class'][1],
                            (nsamples_dict[f'training_samples_per_class'][1] +
                             nsamples_dict[f'validation_samples_per_class'][1]),
                            (nsamples_dict[f'training_samples_per_class'][1] +
                             nsamples_dict[f'validation_samples_per_class'][1] +
                             nsamples_dict[f'testing_samples_per_class'][1])
                            ]


    data_dict['training_noirrig_data'] = noirrig_data_full[0:noirrig_ixs[0]]
    data_dict['validation_noirrig_data'] = noirrig_data_full[noirrig_ixs[0]:noirrig_ixs[1]]
    data_dict['testing_noirrig_data'] = noirrig_data_full[noirrig_ixs[1]:noirrig_ixs[2]]

    data_dict['training_irrig_data'] = irrig_data_full[0:irrig_ixs[0]]
    data_dict['validation_irrig_data'] = irrig_data_full[irrig_ixs[0]:irrig_ixs[1]]
    data_dict['testing_irrig_data'] = irrig_data_full[irrig_ixs[1]:irrig_ixs[2]]


    noirr_clusters_to_toss = input('Which non-irrigated clusters should be discarded? '
                                   '(Enter integers separated by a space or n for none)')

    irr_clusters_to_toss = input('Which irrigated clusters should be discarded? '
                                   '(Enter integers separated by a space or n for none)')

    if noirr_clusters_to_toss == 'n':
        noirr_discard_list = []
    else:
        try:
            noirr_discard_list = [int(i) for i in noirr_clusters_to_toss.split(' ')]
        except Exception as e:
            print(e)
            print('Improper input')

    if irr_clusters_to_toss == 'n':
        irr_discard_list = []
    else:
        try:
            irr_discard_list = [int(i) for i in irr_clusters_to_toss.split(' ')]
        except Exception as e:
            print(e)
            print('Improper input')

    for key, data_array in data_dict.items():
        if '_noirrig_' in key:
            discard_list = noirr_discard_list
        elif '_irrig_' in key:
            discard_list = irr_discard_list
        else:
            print('Bad key')

        data_array = data_array[np.where(~np.isin(data_array[:, -1], discard_list))]
        data_dict[key] = data_array[:, 0:-1]

    print('After cleaning')
    for keys, values in data_dict.items():
        print(f'{keys}: {values.shape}')


    training_arrays_list = []
    validation_arrays_list = []
    testing_arrays_list = []

    for key, values in data_dict.items():
        if 'training' in key:
            training_arrays_list.append(values)
        elif 'validation' in key:
            validation_arrays_list.append(values)
        elif 'testing' in key:
            testing_arrays_list.append(values)

        else:
            print('Invalid key')


    training_array = np.concatenate(training_arrays_list, axis=0)
    validation_array = np.concatenate(validation_arrays_list, axis=0)
    testing_array = np.concatenate(testing_arrays_list, axis=0)

    np.random.seed(7)
    np.random.shuffle(training_array)
    np.random.shuffle(validation_array)
    np.random.shuffle(testing_array)

    print(training_array.shape)
    print(validation_array.shape)
    print(testing_array.shape)

    # out_dir = f'/Volumes/sel_external/ethiopia_irrigation/{region}/training_data/csvs'
    # #
    # # print('Saving files out')
    # pd.DataFrame(training_array).to_csv(f'{out_dir}/{region}_training_labeled_cleaned_pixels.csv')
    # pd.DataFrame(validation_array).to_csv(f'{out_dir}/{region}_validation_labeled_cleaned_pixels.csv')
    # pd.DataFrame(testing_array).to_csv(f'{out_dir}/{region}_testing_labeled_cleaned_pixels.csv')


def clean_csvs_evi_only(region):
    dir_name = f'/Volumes/sel_external/ethiopia_irrigation/{region}/training_data/csvs/evi_only'

    configs = ['training', 'validation', 'testing']

    full_df_list = []
    nsamples_dict = {}

    check_clean = True

    for config in configs:
        nsamples_dict[f'{config}_total_samples'] = 0

        if check_clean:
            csvs = glob(f'{dir_name}/*_{config}_*.csv')
        else:
            csvs = glob(f'{dir_name}/subset_not_cleaned/*_{config}_*.csv')


        config_only_df_list = []

        for csv in csvs:
            df = pd.read_csv(csv, index_col=0)
            full_df_list.append(df)
            config_only_df_list.append(df)

            nsamples_dict[f'{config}_total_samples']+=len(df)

        ## Determine # samples in each classs for each config
        config_only_data = np.array(pd.concat(config_only_df_list))
        config_only_noirrig = config_only_data[np.where(config_only_data[:, -1] == 1)]
        config_only_irrig = config_only_data[np.where(config_only_data[:, -1] == 2)]

        nsamples_dict[f'{config}_samples_per_class'] = [len(config_only_noirrig), len(config_only_irrig)]


    for key, values in nsamples_dict.items():
        print(f'{key}:{values}')

    full_data = np.array(pd.concat(full_df_list))


    noirrig_data_full = full_data[np.where(full_data[:, -1] == 1)]
    irrig_data_full = full_data[np.where(full_data[:, -1] == 2)]

    n_samples = 100000

    noirrig_data = noirrig_data_full[:, 0:-1]
    irrig_data = irrig_data_full[:, 0:-1]


    data_list = [noirrig_data, irrig_data]


    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18,8))
    n_cluster_list = [15, 15]
    n_pca_components = 10
    pred_names = ['Non-irrigated', 'Irrigated']
    ticknames = ['Jun 2020', 'Aug 2020', 'Oct 2020', 'Dec 2020', 'Feb 2021', 'Apr 2021']
    minors = np.linspace(0, 36, 37)

    ylabels_list = []

    for ix, data in enumerate(data_list):
        ax = axes[ix]

        n_clusters = n_cluster_list[ix]

        pca = PCA(n_components=n_pca_components)
        principalComponents = pca.fit_transform(data)

        X = principalComponents
        gmm = GaussianMixture(n_components=n_clusters,
                              covariance_type='full')
        ylabels = gmm.fit_predict(X)
        ylabels_list.append(ylabels)

        for jx in range(n_clusters):
            valid_indices = np.where(ylabels == jx)[0]
            cluster_data = data[valid_indices]

            # ndvi = extract_ndvi(cluster_data)
            ndvi = cluster_data
            ndvi_mean = np.clip(np.mean(ndvi, axis=0), 0, 1)

            if jx >= 10:
                linestyle = '--'
            else:
                linestyle = '-'

            ax.plot(range(len(ndvi_mean)), ndvi_mean, linestyle=linestyle,
                    label = f'Cluster {jx}, {np.count_nonzero(ylabels==jx)} px.')

        if ix == 0:
            ax.legend(loc='upper right')
        else:
            ax.legend(loc='upper left')

        fs = 12
        ax.set_title(f'Total Pixels: {len(data)}', fontsize=fs)
        ax.set_ylabel(f'EVI', fontsize=fs)
        ax.set_xticklabels(ticknames, rotation=40, fontsize=fs)
        ax.xaxis.set_major_locator(IndexLocator(6, 0))
        ax.xaxis.set_minor_locator(FixedLocator(minors))
        ax.tick_params(axis='both', which='major', length=5, labelsize=fs)
        ax.tick_params(axis='x', which='minor', length=3)
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    noirrig_data_full = np.concatenate((noirrig_data_full, ylabels_list[0][..., None]), axis=-1)
    irrig_data_full = np.concatenate((irrig_data_full, ylabels_list[1][..., None]), axis=-1)

    data_dict = {}

    noirrig_ixs = [nsamples_dict[f'training_samples_per_class'][0],
                            (nsamples_dict[f'training_samples_per_class'][0] +
                             nsamples_dict[f'validation_samples_per_class'][0]),
                            (nsamples_dict[f'training_samples_per_class'][0] +
                             nsamples_dict[f'validation_samples_per_class'][0] +
                             nsamples_dict[f'testing_samples_per_class'][0])
                            ]

    irrig_ixs = [nsamples_dict[f'training_samples_per_class'][1],
                            (nsamples_dict[f'training_samples_per_class'][1] +
                             nsamples_dict[f'validation_samples_per_class'][1]),
                            (nsamples_dict[f'training_samples_per_class'][1] +
                             nsamples_dict[f'validation_samples_per_class'][1] +
                             nsamples_dict[f'testing_samples_per_class'][1])
                            ]


    data_dict['training_noirrig_data'] = noirrig_data_full[0:noirrig_ixs[0]]
    data_dict['validation_noirrig_data'] = noirrig_data_full[noirrig_ixs[0]:noirrig_ixs[1]]
    data_dict['testing_noirrig_data'] = noirrig_data_full[noirrig_ixs[1]:noirrig_ixs[2]]

    data_dict['training_irrig_data'] = irrig_data_full[0:irrig_ixs[0]]
    data_dict['validation_irrig_data'] = irrig_data_full[irrig_ixs[0]:irrig_ixs[1]]
    data_dict['testing_irrig_data'] = irrig_data_full[irrig_ixs[1]:irrig_ixs[2]]


    noirr_clusters_to_toss = input('Which non-irrigated clusters should be discarded? '
                                   '(Enter integers separated by a space or n for none)')

    irr_clusters_to_toss = input('Which irrigated clusters should be discarded? '
                                   '(Enter integers separated by a space or n for none)')

    if noirr_clusters_to_toss == 'n':
        noirr_discard_list = []
    else:
        try:
            noirr_discard_list = [int(i) for i in noirr_clusters_to_toss.split(' ')]
        except Exception as e:
            print(e)
            print('Improper input')

    if irr_clusters_to_toss == 'n':
        irr_discard_list = []
    else:
        try:
            irr_discard_list = [int(i) for i in irr_clusters_to_toss.split(' ')]
        except Exception as e:
            print(e)
            print('Improper input')

    for key, data_array in data_dict.items():
        if '_noirrig_' in key:
            discard_list = noirr_discard_list
        elif '_irrig_' in key:
            discard_list = irr_discard_list
        else:
            print('Bad key')

        data_array = data_array[np.where(~np.isin(data_array[:, -1], discard_list))]
        data_dict[key] = data_array[:, 0:-1]

    print('After cleaning')
    for keys, values in data_dict.items():
        print(f'{keys}: {values.shape}')


    training_arrays_list = []
    validation_arrays_list = []
    testing_arrays_list = []

    for key, values in data_dict.items():
        if 'training' in key:
            training_arrays_list.append(values)
        elif 'validation' in key:
            validation_arrays_list.append(values)
        elif 'testing' in key:
            testing_arrays_list.append(values)

        else:
            print('Invalid key')


    training_array = np.concatenate(training_arrays_list, axis=0)
    validation_array = np.concatenate(validation_arrays_list, axis=0)
    testing_array = np.concatenate(testing_arrays_list, axis=0)

    np.random.seed(7)
    np.random.shuffle(training_array)
    np.random.shuffle(validation_array)
    np.random.shuffle(testing_array)

    print(training_array.shape)
    print(validation_array.shape)
    print(testing_array.shape)

    #
    # print('Saving files out')
    pd.DataFrame(training_array).to_csv(f'{dir_name}/{region}_training_labeled_cleaned_pixels.csv')
    pd.DataFrame(validation_array).to_csv(f'{dir_name}/{region}_validation_labeled_cleaned_pixels.csv')
    pd.DataFrame(testing_array).to_csv(f'{dir_name}/{region}_testing_labeled_cleaned_pixels.csv')


def clean_inference_samples_evi_only():
    dir_name = f'/Volumes/sel_external/ethiopia_irrigation/inference_assesment/training_data/csvs/evi_only'



    check_clean = True


    if check_clean:
        csvs = glob(f'{dir_name}/cluster_cleaned_and_filtered/*.csv')
    else:
        csvs = glob(f'{dir_name}/not_cleaned/*.csv')

    print(csvs)

    df_list = []

    for kx, csv in enumerate(csvs):
        print(f'Cleaning csv: {csv}')

        csv_ix = csv.split('_')[-1].strip('.csv')

        full_data = np.array(pd.read_csv(csv, index_col=0))


        noirrig_data_full = full_data[np.where(full_data[:, -1] == 1)]
        irrig_data_full = full_data[np.where(full_data[:, -1] == 2)]

        noirrig_data = noirrig_data_full[:, 0:-1]/10000
        irrig_data = irrig_data_full[:, 0:-1]/10000

        print(f'Total non-irrigated samples: {len(noirrig_data)}')
        print(f'Total irrigated samples: {len(irrig_data)}')

        data_list = [noirrig_data, irrig_data]

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))
        n_cluster_list = [20, 20]
        n_pca_components = 10
        pred_names = ['Non-irrigated', 'Irrigated']
        ticknames = ['Jun 2020', 'Aug 2020', 'Oct 2020', 'Dec 2020', 'Feb 2021', 'Apr 2021']
        minors = np.linspace(0, 36, 37)

        ylabels_list = []

        for ix, data in enumerate(data_list):
            ax = axes[ix]

            n_clusters = n_cluster_list[ix]

            print('PCA transform')
            pca = PCA(n_components=n_pca_components)
            principalComponents = pca.fit_transform(data)

            print('GMM fitting')

            X = principalComponents
            gmm = GaussianMixture(n_components=n_clusters,
                                  covariance_type='full')
            ylabels = gmm.fit_predict(X)
            ylabels_list.append(ylabels)

            for jx in range(n_clusters):
                valid_indices = np.where(ylabels == jx)[0]
                cluster_data = data[valid_indices]

                # ndvi = extract_ndvi(cluster_data)
                ndvi = cluster_data
                ndvi_mean = np.clip(np.mean(ndvi, axis=0), 0, 1)

                if jx >= 10:
                    linestyle = '--'
                else:
                    linestyle = '-'

                ax.plot(range(len(ndvi_mean)), ndvi_mean, linestyle=linestyle,
                        label=f'Cluster {jx}, {np.count_nonzero(ylabels == jx)} px.')

            if ix == 0:
                ax.legend(loc='upper center')
            else:
                ax.legend(loc='upper left')

            fs = 12
            ax.set_title(f'Total Pixels: {len(data)}', fontsize=fs)
            ax.set_ylabel(f'EVI', fontsize=fs)
            ax.set_xticklabels(ticknames, rotation=40, fontsize=fs)
            ax.xaxis.set_major_locator(IndexLocator(6, 0))
            ax.xaxis.set_minor_locator(FixedLocator(minors))
            ax.tick_params(axis='both', which='major', length=5, labelsize=fs)
            ax.tick_params(axis='x', which='minor', length=3)
            ax.grid(True)

        plt.tight_layout()
        plt.show()

        noirrig_data_full = np.concatenate((noirrig_data_full, ylabels_list[0][..., None]), axis=-1)
        irrig_data_full = np.concatenate((irrig_data_full, ylabels_list[1][..., None]), axis=-1)

        data_dict = {}


        data_dict['testing_noirrig_data'] = noirrig_data_full
        data_dict['testing_irrig_data']   = irrig_data_full

        noirr_clusters_to_toss = input('Which non-irrigated clusters should be discarded? '
                                       '(Enter integers separated by a space or n for none)')

        irr_clusters_to_toss = input('Which irrigated clusters should be discarded? '
                                     '(Enter integers separated by a space or n for none)')

        if noirr_clusters_to_toss == 'n':
            noirr_discard_list = []
        else:
            try:
                noirr_discard_list = [int(i) for i in noirr_clusters_to_toss.split(' ')]
            except Exception as e:
                print(e)
                print('Improper input')

        if irr_clusters_to_toss == 'n':
            irr_discard_list = []
        else:
            try:
                irr_discard_list = [int(i) for i in irr_clusters_to_toss.split(' ')]
            except Exception as e:
                print(e)
                print('Improper input')

        for key, data_array in data_dict.items():
            if '_noirrig_' in key:
                discard_list = noirr_discard_list
            elif '_irrig_' in key:
                discard_list = irr_discard_list
            else:
                print('Bad key')

            data_array = data_array[np.where(~np.isin(data_array[:, -1], discard_list))]
            data_dict[key] = data_array[:, 0:-1]

        print('After cleaning')
        arrays_list = []

        for keys, values in data_dict.items():
            print(f'{keys}: {values.shape}')
            arrays_list.append(values)

        out_array = np.concatenate(arrays_list, axis=0)
        np.random.shuffle(out_array)
        pd.DataFrame(out_array).to_csv(f'{dir_name}/amhara_testing_labeled_cleaned_pixels_{csv_ix}.csv')


def filter_out_invalid_irrig_samples():
    dir_name = f'/Volumes/sel_external/ethiopia_irrigation/inference_assesment/training_data/csvs/evi_only'

    csvs = glob(f'{dir_name}/cluster_cleaned_only/*.csv')
    print(csvs)

    df_list = []

    for kx, csv in enumerate(csvs):
        print(f'Cleaning csv: {csv}')

        full_data = np.array(pd.read_csv(csv, index_col=0))

        noirrig_data_full = full_data[np.where(full_data[:, -1] == 1)]
        irrig_data_full = full_data[np.where(full_data[:, -1] == 2)]
        irrig_data = irrig_data_full[:, 0:-1] / 10000

        print(f'Shape of original irrigation data: {irrig_data.shape}')

        min_vi = np.percentile(irrig_data, q=10, axis=-1) + np.finfo(float).eps
        max_vi = np.percentile(irrig_data, q=90, axis=-1)

        vi_ratio_boolean = ((max_vi / min_vi) >= 2).astype(np.int16)

        min_vi_boolean = (min_vi <= 0.2).astype(np.int16)
        max_vi_boolean = (max_vi >= 0.2).astype(np.int16)

        dry_season_start_ix = 18
        dry_season_end_ix = 30

        dry_season_s2 = irrig_data[:, dry_season_start_ix:dry_season_end_ix]
        dry_season_max_vi = np.max(dry_season_s2, axis=-1)
        valid_dry_season = (dry_season_max_vi >= 0.2).astype(np.int16)

        out_image = np.stack((min_vi_boolean, max_vi_boolean, vi_ratio_boolean, valid_dry_season), axis=-1)

        out_irrig = np.min(out_image, axis=-1).astype(np.int16)
        print(f'Fraction of pixels being kept: {np.mean(out_irrig)}')

        irrig_data_filtered = irrig_data_full[np.where(out_irrig)]

        print(f'Shape of filtered data: {irrig_data_filtered.shape}')

        total_data = np.concatenate((noirrig_data_full, irrig_data_filtered), axis=0)
        print(f'Shape of recombined data: {total_data.shape}')

        file_ext = csv.split('/')[-1]
        file_name = f'{dir_name}/cluster_cleaned_and_filtered/{file_ext}'

        pd.DataFrame(total_data).to_csv(file_name)

if __name__ == '__main__':
    # # regions = []
    # regions = ['tana', 'rift',  'koga', 'kobo', 'alamata', 'liben', 'jiga', 'motta']
    #
    #
    region = 'tana'
    clean_csvs(region)
    # clean_csvs_evi_only(region)

