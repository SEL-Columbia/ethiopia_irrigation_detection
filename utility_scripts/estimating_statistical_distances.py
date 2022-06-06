import numpy as np
from glob import glob
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import itertools
import pickle as pk
import matplotlib.pyplot as plt
from matplotlib.ticker import  FixedLocator, IndexLocator
import seaborn as sns

colors_xkcd = ['very dark purple', "windows blue", "amber", "greyish",
                   "faded green", 'pastel blue', 'grape', 'dark turquoise', 'terracotta',
                   'salmon pink', 'evergreen'
                   ]
cmap = sns.color_palette(sns.xkcd_palette(colors_xkcd), desat=1)
sns.set_palette(sns.xkcd_palette(colors_xkcd))
sns.set_style("whitegrid")

def determine_pca_transform_for_all_data():
    regions = ['tana', 'rift', 'koga', 'kobo', 'alamata', 'liben', 'jiga', 'motta']

    features, labels = load_data(regions, 'training')

    n_pca_components = 10

    pca = PCA(n_components=n_pca_components)
    principalComponents = pca.fit_transform(features)

    save_dir = f"../data/saved_pca_and_gmm_models/all_regions"
    # pk.dump(pca, open(f"{save_dir}/pca_all_regions_100k_per_region_{pca.n_components}_ncomponents.pkl", "wb"))


# def bhattacharyya_gaussian_distance(distribution1: "dict", distribution2: "dict", ) -> int:
def bhattacharyya_gaussian_distance(mean1, cov1, mean2, cov2) -> int:
    """ Estimate Bhattacharyya Distance (between Gaussian Distributions)

    Args:
        distribution1: a sample gaussian distribution 1
        distribution2: a sample gaussian distribution 2

    Returns:
        Bhattacharyya distance
    """
    # mean1 = distribution1["mean"]
    # cov1 = distribution1["covariance"]
    #
    # mean2 = distribution2["mean"]
    # cov2 = distribution2["covariance"]

    cov = (1 / 2) * (cov1 + cov2)

    T1 = (1 / 8) * (
        np.sqrt((mean1 - mean2) @ np.linalg.inv(cov) @ (mean1 - mean2).T)[0][0]
    )
    T2 = (1 / 2) * np.log(
        np.linalg.det(cov) / np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2))
    )

    return T1 + T2

def load_data(regions, config):

    # config = 'validat'

    data_list = []
    labels_list = []

    bands_to_select = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]



    if config == 'training':
        n_samples = 100000
    elif config == 'validation' or config == 'testing':
        n_samples = 30000


    for region in regions:
        print(f'Loading data, region: {region}')

        csv_dir = f'/Volumes/sel_external/ethiopia_irrigation/{region}/training_data/csvs'
        csv_files = glob(f'{csv_dir}/*{config}*.csv')

        regional_data_list = []
        regional_labels_list = []

        for file in tqdm(csv_files):

            df = pd.read_csv(file, index_col=0)
            data = np.array(df)
            features = data[:, 5*13:41*13]

            features = np.reshape(features, (len(features), 36, 13))
            features = features[:, :, bands_to_select]

            features = np.reshape(features, (features.shape[0], features.shape[1] * features.shape[2]))

            labels = data[:, -1]

            regional_data_list.append(features)
            regional_labels_list.append(labels)

        features = np.concatenate(regional_data_list, axis=0)
        labels = np.concatenate(regional_labels_list, axis=0)

        p = np.random.permutation(len(features))
        features = features[p][0:n_samples]
        labels = labels[p][0:n_samples]

        print(f'Num samples for region {region}: {len(features)}')

        data_list.append(features)
        labels_list.append(labels)

    features = np.concatenate(data_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    p = np.random.permutation(len(features))

    return features[p], labels[p]


def save_gmm_cluster_data(regions, config):

    save_dir = f"../data/saved_pca_and_gmm_models/all_regions"
    pca = pk.load(open(f"{save_dir}/pca_all_regions_100k_per_region_10_ncomponents.pkl", "rb"))

    class_strings = ['noirrig', 'irrig']
    regional_dict = {}

    for region in regions:
        print(f'Loading data for region: {region}')
        full_features, labels = load_data([region], config)
        non_irrig = full_features[labels == 1]
        irrig = full_features[labels == 2]

        comb_data = [non_irrig, irrig]

        for ix, features in enumerate(comb_data):
            print('Applying PCA')
            principalComponents = pca.transform(features)

            print('Apply GMM')
            gmm = GaussianMixture(n_components=1,
                                  covariance_type='full',
                                  )

            gmm.fit(principalComponents)
            print(f'{class_strings[ix]} lb : {gmm.lower_bound_}')

            regional_dict[f'{region}_{class_strings[ix]}_mean'] = gmm.means_
            regional_dict[f'{region}_{class_strings[ix]}_covariance'] = gmm.covariances_

    # save_dir = f"../data/saved_pca_and_gmm_models/all_regions"
    # pk.dump(regional_dict, open(f"{save_dir}/gmm_stats_all_regions_by_class.pkl", "wb"))

    for key, value in regional_dict.items():
        print(f'{key}:{value}')

def determine_gaussian_distance(regions):

    irrig_distance_matrix = np.zeros((len(regions), len(regions)))
    noirrig_distance_matrix = np.zeros((len(regions), len(regions)))

    save_dir = f"../data/saved_pca_and_gmm_models/all_regions"
    dt = pk.load(open(f"{save_dir}/gmm_stats_all_regions_by_class.pkl", "rb"))


    for ix, region_a in enumerate(regions):
        print(f'Calculating Gaussian distances for {region_a}')

        for jx, region_b in enumerate(regions):

            irrig_distance = bhattacharyya_gaussian_distance(dt[f'{region_a}_irrig_mean'],
                                                             dt[f'{region_a}_irrig_covariance'],
                                                             dt[f'{region_b}_irrig_mean'],
                                                             dt[f'{region_b}_irrig_covariance'],
                                                             )

            noirrig_distance = bhattacharyya_gaussian_distance(dt[f'{region_a}_noirrig_mean'],
                                                             dt[f'{region_a}_noirrig_covariance'],
                                                             dt[f'{region_b}_noirrig_mean'],
                                                             dt[f'{region_b}_noirrig_covariance'],
                                                             )

            irrig_distance_matrix[ix, jx] = irrig_distance
            noirrig_distance_matrix[ix, jx] = noirrig_distance

    out_dir = '../results/bhattacharyya_distance_results'
    pd.DataFrame(irrig_distance_matrix, columns=regions, index=regions).to_csv(f'{out_dir}/all_regions_irrig.csv')
    pd.DataFrame(noirrig_distance_matrix, columns=regions, index=regions).to_csv(f'{out_dir}/all_regions_noirrig.csv')


def plot_cov_norms(regions):
    irrig_l2_norm = np.zeros((len(regions)))
    noirrig_l2_norm = np.zeros((len(regions)))

    save_dir = f"../data/saved_pca_and_gmm_models/all_regions"
    dt = pk.load(open(f"{save_dir}/gmm_stats_all_regions_by_class.pkl", "rb"))

    for key, value in dt.items():
        print(f'{key}: {value}')

    for ix, region_a in enumerate(regions):
        print(f'Calculating Gaussian distances for {region_a}')

        irrig_cov = dt[f'{region_a}_irrig_covariance']
        noirrig_cov = dt[f'{region_a}_noirrig_covariance']


        irrig_l2_norm[ix] = np.linalg.norm(irrig_cov[0], 2)
        print(f'Irrig l2: {irrig_l2_norm[ix]}')


        noirrig_l2_norm[ix] = np.linalg.norm(noirrig_cov[0], 2)
        print(f'noirrig l2: {noirrig_l2_norm[ix]}')


    min_val = np.min((irrig_l2_norm, noirrig_l2_norm))
    max_val = np.max((irrig_l2_norm, noirrig_l2_norm))

    fig, ax = plt.subplots()



    ax.scatter(irrig_l2_norm, noirrig_l2_norm)

    for ix, region in enumerate(regions):
        ax.annotate(region, (irrig_l2_norm[ix]*1.05, noirrig_l2_norm[ix]*1.02))

    ax.plot(np.linspace(min_val, max_val, 100), np.linspace(min_val, max_val, 100))

    ax.set_xlabel('L2 Norm of Irrigated Samples\' Covariance' )
    ax.set_ylabel('L2 Norm of Non-Irrigated Samples\' Covariance')

    plt.show()



if __name__ == '__main__':
    config = 'training'
    regions = ['tana', 'rift', 'koga', 'kobo', 'alamata', 'liben', 'jiga', 'motta']

    plot_cov_norms(regions)
