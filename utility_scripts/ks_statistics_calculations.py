import numpy as np
from glob import glob
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import itertools
import pickle as pk
import matplotlib.pyplot as plt
from matplotlib.ticker import (FixedLocator, IndexLocator)
from scipy.stats import normaltest
from scipy.stats import kstest, wasserstein_distance
from scipy.spatial.distance import jensenshannon



def load_data(regions):

    config = 'training'
    dfs = []
    # print('Loadings CSVs')
    np.random.seed(7)

    for region in regions:
        print(f'Loading data for region: {region}')
        csv_dir = f'/Volumes/sel_external/ethiopia_irrigation/{region}/training_data/csvs/evi_only/'

        csvs = glob(f'{csv_dir}/*{config}*.csv')
        np.random.shuffle(csvs)
        print(len(csvs))

        # indices = list(np.arange(5*13,41*13)) + [-1]

        for csv in tqdm(csvs):

            data = pd.read_csv(csv, index_col=0)
            dfs.append(data)

    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df[(full_df.T != 0).any()]

    data = np.array(full_df.values)

    num_samples = {}
    num_samples['irrigated'] = np.count_nonzero(data[:, -1] == 2)
    num_samples['non-irrigated'] = np.count_nonzero(data[:, -1] == 1)

    # print(f'Num non-irrigated pixels: {num_samples["non-irrigated"]}')
    # print(f'Num irrigated pixels: {num_samples["irrigated"]}')

    print(f'Full data shape: {data.shape}')

    return data


def extract_evi(data):

    data = np.reshape(data, (data.shape[0], 36, 13))
    evi = (data[..., 7] - data[..., 3]) / (data[..., 7] + 6*data[..., 3] - 7.5*data[..., 1] + 10000)

    print(f'EVI shape: {data.shape}')


    return evi


def pca_transform(img_array):
    # pca_transform takes an array of pixel timeseries and reduces the dimensionality

    print('Determine PCA transform')
    n_components = 8
    min_px = 1500000

    shuffle = True
    if shuffle:
        np.random.seed(7)
        np.random.shuffle(img_array)

    print(img_array.shape)

    # Only take non-zero  timeseries
    max_pixels = np.min((min_px, len(img_array)))
    print(max_pixels)


    full_data = img_array[0:max_pixels]

    num_samples = {}
    num_samples['irrigated'] = np.count_nonzero(full_data[:, -1] == 2)
    num_samples['non-irrigated'] = np.count_nonzero(full_data[:, -1] == 1)

    print(f'Num non-irrigated pixels: {num_samples["non-irrigated"]}')
    print(f'Num irrigated pixels: {num_samples["irrigated"]}')

    # data = full_data[:, 5*13:41*13]
    labels = full_data[:, -1]
    data = full_data[:, 0:-1]

    # Extract EVI
    # data = extract_evi(data)

    # Initialize a PCA model and fit to the data
    pca = PCA(n_components= n_components)
    principalComponents = pca.fit_transform(data)

    exp_var = pca.explained_variance_
    cum_var = np.zeros(len(exp_var))
    sum_var = np.sum(pca.explained_variance_)

    for ix in range(1, len(cum_var)+1):
        cum_var[ix-1] = np.sum(exp_var[0:ix])



    print(f'Explained variance: {pca.explained_variance_}')
    print(f'Total variance: {sum_var}')
    print(f'Cumulative variance: {cum_var/sum_var}')

    save = True
    region = 'all_regions'
    layer = 'evi'

    if save:
        save_dir = f"../data/saved_pca_and_gmm_models/{region}/{layer}"
        pk.dump(pca, open(f"{save_dir}/pca_{region}_ncomponents_{pca.n_components}_"
                          f"varexplained_{(cum_var[-1]/sum_var):3f}.pkl", "wb"))

    return principalComponents, pca, data, num_samples, labels


def plot_pca_stats():
    region = 'all_regions'
    layer = 'evi'

    save_dir = f"../data/saved_pca_and_gmm_models/{region}/{layer}"
    pca = pk.load(open(f"{save_dir}/pca_{region}_ncomponents_35_"
                          f"varexplained_1.000.pkl", "rb"))

    fig, ax = plt.subplots()
    ax.plot(range(35), pca.explained_variance_, label = 'All labeled data (7VV + 1GV regions)')
    ax.set_xlabel('PCA Dimension')
    ax.set_ylabel('Explained Variance')
    ax.grid(True)
    ax.legend()

    ticknames = ['Jun 2020', 'Aug 2020', 'Oct 2020', 'Dec 2020', 'Feb 2021', 'Apr 2021']


    n_eofs = 10
    fig1, ax1 = plt.subplots(nrows=n_eofs, ncols=1, figsize = (9,9))
    for ix in range(n_eofs):
        ax1[ix].plot(range(36), pca.components_[ix])
        ax1[ix].set_ylabel(f'EOF {ix}')
        ax1[ix].grid(True)

        ax1[ix].xaxis.set_major_locator(IndexLocator(6, 0))
        ax1[ix].xaxis.set_minor_locator(IndexLocator(1, 0))
        ax1[ix].tick_params(axis='x', which='both', length=2)

        if ix == n_eofs - 1:
            ax1[ix].set_xticklabels(ticknames, rotation=40)
        else:
            ax1[ix].set_xticklabels('')

    fig.tight_layout()
    fig1.tight_layout()
    plt.show()


def determine_if_gaussian(region):

    full_data = load_data([region])

    shuffle = True
    if shuffle:
        np.random.seed(7)
        np.random.shuffle(full_data)



    labels = full_data[:, -1]
    data = full_data[:, 0:-1]

    evi = extract_evi(data)

    region = 'all_regions'
    layer = 'evi'
    save_dir = f"../data/saved_pca_and_gmm_models/{region}/{layer}"

    pca = pk.load(open(f"{save_dir}/pca_{region}_ncomponents_35_"
                          f"varexplained_1.000.pkl", "rb"))



    transformed_data = pca.transform(evi)

    class_strings = ['no-irrig', 'irrig']

    for ix in range(2):
        indices = np.where(labels == ix+1)
        X = transformed_data[indices]

        min_px = 200000
        max_pixels = np.min((min_px, len(X)))
        full_data = X[0:max_pixels]

        for jx in range(3):
            out_str = f'Region {region}, {class_strings[ix]}, PC Dim {jx}'
            print('Normal test')
            stat, p = normaltest(full_data[jx])
            print('stat=%.3f, p=%.3f' % (stat, p))
            if p > 0.05:
                print(f'{out_str}: Probably Gaussian')
            else:
                print(f'{out_str}: Probably not Gaussian')

        print('-------------')


def calculate_ks_distance(regions, class_name):

    region = 'all_regions'
    layer = 'evi'
    save_dir = f"../data/saved_pca_and_gmm_models/{region}/{layer}"

    # pca = pk.load(open(f"{save_dir}/pca_{region}_ncomponents_8_"
    #                       f"varexplained_1.000.pkl", "rb"))

    if class_name == 'noirrig':
        class_ix = 0
    else:
        class_ix = 1

    out_array = np.zeros((len(regions), len(regions)))
    n_pca_dims = 8
    np.random.seed(7)


    for ix, region1 in enumerate(regions):
        print(f'Region1: {region1}')
        full_data1 = load_data([region1])
        np.random.shuffle(full_data1)

        if ix == 0:
            print(full_data1[0])

        labels1 = full_data1[:, -1]
        data1 = full_data1[:, 0:-1]

        indices1 = np.where(labels1 == class_ix+1)
        data1 = data1[indices1]

        # evi1 = extract_evi(data1)
        evi1_transformed = data1 #pca.transform(data1)


        for jx in range(1, len(regions)-ix):
            region2 = regions[ix + jx]
            print(f'Region2: {region2}')

            full_data2 = load_data([region2])
            np.random.shuffle(full_data2)

            labels2 = full_data2[:, -1]
            data2 = full_data2[:, 0:-1]

            indices2 = np.where(labels2 == class_ix+1)
            data2 = data2[indices2]

            # evi2 = extract_evi(data2)
            evi2_transformed = data2 # pca.transform(data2)

            ks_stats = np.zeros(36)

            min_samples = np.min((evi1_transformed.shape[0], evi2_transformed.shape[0]))
            # print(min_samples)

            for kx in range(n_pca_dims):
                p = evi1_transformed[:, kx]
                q = evi2_transformed[:, kx]

                js, p_score = kstest(p, q)
                # js, p_score = kstest(p, q)
                ks_stats[kx] = js

            ks_geom_dist = np.sqrt(np.sum(np.square(ks_stats)))

            print(f'KS GEOM Distance: {ks_geom_dist}')

            out_array[ix, ix+jx] = ks_geom_dist
            out_array[ix+jx, ix] = ks_geom_dist


    print(out_array)
    df = pd.DataFrame(out_array, columns=regions, index=regions)
    df.to_csv(f'../data/cluster_distances/pseudo_1dks_pcadims_36_all_{class_name}_samples_.csv')






if __name__ == '__main__':

    regions = ['tana', 'rift',  'koga', 'kobo', 'alamata', 'liben', 'jiga', 'motta']
    # data = load_data(regions)
    # principalComponents, pca, data, num_samples, labels = pca_transform(data)

    calculate_ks_distance(regions, 'noirrig')

