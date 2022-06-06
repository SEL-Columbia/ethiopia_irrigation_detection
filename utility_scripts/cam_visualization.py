import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from learning.model import *
import tensorflow.keras as keras
import sklearn
from learning.dataloader import DataGenerator
from scipy.interpolate import interp1d
import datetime
from learning.training_main import dotdict, get_args
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               FixedLocator, IndexLocator, LinearLocator)
import pandas as pd
from tqdm import tqdm
from keras_applications.resnet import ResNet50
from sklearn.utils import shuffle

colors_xkcd = ['very dark purple', "windows blue", "amber", "greyish",
               "faded green", 'pastel blue']
cmap = sns.color_palette(sns.xkcd_palette(colors_xkcd), desat=1)
sns.set_palette(sns.xkcd_palette(colors_xkcd))

def calculate_evi(generator, region, ts):

    if len(ts.shape) == 3:
        ts = ts[0]

    # Retrieve non-normalized values
    band_means = np.array(generator.norm_df[f'{region}_mean'])
    band_std = np.array(generator.norm_df[f'{region}_std'])

    band_means = np.reshape(band_means, (ts.shape[0], ts.shape[1]-1))
    band_std = np.reshape(band_std, (ts.shape[0], ts.shape[1]-1))

    ts_denorm = np.zeros((ts.shape[0], ts.shape[1]-1))

    for ix in range(ts_denorm.shape[1]-1):
        ts_denorm[:, ix] = ts[:, ix] * band_std[:, ix] + band_means[:, ix]

    rainfall = ts[:, -1]

    # Calculate EVI
    L = 1
    C1 = 6
    C2 = 7.5
    G = 2.5

    NIR = ts_denorm[:, 7]/10000
    RED = ts_denorm[:, 3]/10000
    BLUE = ts_denorm[:, 1]/10000


    evi = G * (NIR-RED)/(NIR + C1*RED - C2*BLUE + L)
    # evi = (NIR-RED)/(NIR + RED)


    return evi, rainfall


def calculate_ndvi(generator, region, ts):

    if len(ts.shape) == 3:
        ts = ts[0]

    # Retrieve non-normalized values
    band_means = np.array(generator.norm_df[f'{region}_mean'])
    band_std = np.array(generator.norm_df[f'{region}_std'])

    band_means = np.reshape(band_means, (ts.shape[0], ts.shape[1]-1))
    band_std = np.reshape(band_std, (ts.shape[0], ts.shape[1]-1))

    ts_denorm = np.zeros((ts.shape[0], ts.shape[1]-1))

    for ix in range(ts_denorm.shape[1]-1):
        ts_denorm[:, ix] = ts[:, ix] * band_std[:, ix] + band_means[:, ix]

    rainfall = ts[:, -1]

    ndvi = (ts_denorm[..., 6] - ts_denorm[..., 2]) / (ts_denorm[..., 6] + ts_denorm[..., 2])

    return ndvi, rainfall


def load_test_ds(region):

    max_length = None


    csv = f'/Volumes/sel_external/ethiopia_irrigation/{region}/training_data/csvs/evi_only/' \
          f'{region}_testing_evi_only_labeled_cleaned_pixels.csv'

    full_data = np.array(pd.read_csv(csv, index_col=0).values)
    data = full_data[:, 0:36]
    labels = full_data[:, -1]

    if max_length is not None:
        data = data[0:max_length]
        labels = labels[0:max_length]


    # data = (data - 32638) / 32638

    data = np.clip(data, 0, 1)
    print(data[0:2])

    ## Normalize
    data = (data - 0.2555) / 0.16886
    # ds = tf.data.Dataset.from_tensor_slices(data).batch(256)

    return data, labels


def viz_cam():
    ticknames = ['Jun 2020', 'Aug 2020', 'Oct 2020', 'Dec 2020', 'Feb 2021', 'Apr 2021']
    minors = np.linspace(0, 36, 37)

    ## Shifted model: 20211024-152947
    ## Non-shifted model: 20211024-172013


    transformer = f'/Users/terenceconlon/Documents/Columbia - Summer 2021/' \
                  f'ethiopia_irrigation_detection/data/from_gcp/trained_models/20211024-172013'

    lr = 1e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    region = 'koga'
    config = 'testing'

    print('Loading data')
    data, labels = load_test_ds(region)
    print(f'data shape: {data.shape}')

    shuffle_ix = 9
    data, labels = shuffle(data, labels, random_state=shuffle_ix)


    x_train = data[..., None]
    y_train = labels - 1


    classes = np.unique(y_train)


    print('Loading model')

    model = build_model(
        input_shape=(36,1),
        head_size=64,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[],
        mlp_dropout=0,
        dropout=0,
    )

    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer,
        model=model,
    )


    status = checkpoint.restore(tf.train.latest_checkpoint(transformer)).expect_partial()
    wts = model.layers[-1].weights


    # filters
    w_k_c = model.layers[-1].get_weights()[0]  # weights for each filter k for each class c
    print(f'WKC: {w_k_c.shape}')

    # the same input
    new_input_layer = model.inputs
    # output is both the original as well as the before last layer
    new_output_layer = [model.layers[-3].output, model.layers[-1].output]


    new_feed_forward = keras.backend.function(new_input_layer, new_output_layer)

    class_names = ['Non-Irrigated', 'Irrigated']

    for c in classes:
        # plt.figure()
        c = int(c)

        count = 0
        c_x_train = x_train[np.where(y_train == c)]

        if c == 0:
            class_label = 'Non-Irrigated'
        else:
            class_label = 'Irrigated'

        print(f'Number of timeseries in {class_label}: {c_x_train.shape[0]}')
        # print(c_x_train[0].shape)
        pbar = tqdm(total=c_x_train.shape[0], position=0, leave=True)

        predicted_list = []

        print(c)
        fig, ax = plt.subplots(figsize=(11,7))

        max_ts = 16
        ts_counter = 0

        for jx, ts in enumerate(c_x_train):
            if ts_counter == max_ts:
                break

            pbar.update(1)

            ts = ts.reshape(1, -1, 1)

            [conv_out, predicted] = new_feed_forward([ts])


            predicted_list.append(np.argmax(predicted[0]))


            pred_label = np.array(int(predicted[0][1] >= 0.5))
            orig_label = int(c)


            if pred_label == orig_label:
                cas = np.zeros(dtype=np.float, shape=(conv_out.shape[1]))
                out_cas = np.zeros(dtype=np.float, shape=(conv_out.shape[1]))

                # cas = w_k_c[:, orig_label] * conv_out[0]

                for k, w in enumerate(w_k_c[:, orig_label]):

                    cas[k] = w * conv_out[0, k]

                for k, w in enumerate(w_k_c[:, 1]):
                    out_cas[k] = w * conv_out[0, k]

                minimum = np.min(cas)

                cas = cas - minimum

                cas = cas / max(cas)
                cas = cas * 100

                max_length = 10000

                # Unnormalize
                ts = ts * 0.16886 + 0.2555

                x = np.linspace(0, ts.shape[1] - 1, max_length, endpoint=True)
                # linear interpolation to smooth
                f = interp1d(range(ts.shape[1]), ts[0, :, 0])
                y = f(x)

                f = interp1d(range(ts.shape[1]), cas)
                cas = f(x).astype(int)
                ts_counter += 1

                im = ax.scatter(x=x, y=y, c=cas, cmap='jet', marker='.', s=8, vmin=0, vmax=100, linewidths=0.0)


        fs = 16

        # cbar = fig.colorbar(im, pad=0.1)
        ax.set_xticklabels(ticknames, rotation=40)
        ax.xaxis.set_major_locator(IndexLocator(6, 0))
        ax.xaxis.set_minor_locator(IndexLocator(1, 0))
        ax.tick_params(axis='both', which='both', length=5, labelsize=fs)
        ax.tick_params(axis='x', which='minor', length=3)
        ax.grid(True)
        # ax.set_title(f'{region.capitalize()}, {class_label},\n16 Sample EVI Timeseries and GradCAM Importances')
        ax.set_ylabel(f'EVI', fontsize=fs)
        # cbar.ax.set_title('Normalized Logit\nImportance')
        # cbar.ax.tick_params(length=3, labelsize=12)

        fig.tight_layout()

        plt.show()


def plotting():
    ## Load cam files

    max_length = 500
    ticknames = ['Jun 2020', 'Aug 2020', 'Oct 2020', 'Dec 2020', 'Feb 2021', 'Apr 2021', 'Jun 2021']
    minors = np.linspace(0, 35, 36)

    px_class = 0

    base_cam = np.array(pd.read_csv(f'../data/cam_files/tana_testing_baseline_cam_class_{px_class}.csv',
                                    index_col=0))
    trans_cam = np.array(pd.read_csv(f'../data/cam_files/tana_testing_transformer_cam_class_{px_class}.csv',
                                     index_col=0))

    print(len(base_cam))

    ndvi_0 = np.array(pd.read_csv('../data/cam_files/tana_testing_transformer_ndvi_pred_class_0.csv',
                                     index_col=0))
    ndvi_1 = np.array(pd.read_csv('../data/cam_files/tana_testing_transformer_ndvi_pred_class_1.csv',
                                     index_col=0))

    ndvi_mean_0 = np.mean(ndvi_0[:, 0:36], axis=0)
    ndvi_mean_1 = np.mean(ndvi_1[:, 0:36], axis=0)



    base_cam_mean = np.mean(base_cam, axis=0)
    trans_cam_mean = np.mean(trans_cam, axis=0)

    sns.heatmap(trans_cam[0:500])

    fig, ax = plt.subplots()
    ax.plot(range(base_cam_mean.shape[-1]), base_cam_mean, label='Baseline')
    ax.plot(range(trans_cam_mean.shape[-1]), trans_cam_mean, label='Transformer')

    ax.grid(True)
    ax.xaxis.set_major_locator(IndexLocator(6, 0))
    ax.xaxis.set_minor_locator(FixedLocator(minors))
    ax.set_xticklabels(ticknames, rotation=40)
    ax.tick_params(axis='x', which='major', length=5)
    ax.tick_params(axis='x', which='minor', length=3)
    ax.set_ylabel('Mean Normalized Logit Importance, Tana Test Set')

    ax.legend()


    fig1, ax1 = plt.subplots()
    ax1.plot(range(base_cam_mean.shape[-1]), ndvi_mean_0, label='Non-Irrigated')
    ax1.plot(range(base_cam_mean.shape[-1]), ndvi_mean_1, label='Irrigated')

    ax1.grid(True)
    ax1.xaxis.set_major_locator(IndexLocator(6, 0))
    ax1.xaxis.set_minor_locator(FixedLocator(minors))
    ax1.set_xticklabels(ticknames, rotation=40)
    ax1.tick_params(axis='x', which='major', length=5)
    ax1.tick_params(axis='x', which='minor', length=3)
    ax1.set_ylabel('NDVI')
    ax1.legend()

    plt.show()




    # im = ax.scatter(x=x, y=y, c=cas, cmap='jet', marker='.', s=10, vmin=0, vmax=100, linewidths=0.0)
    # cbar = fig.colorbar(im, pad=0.12)
    #
    # ax01.plot(range(ts.shape[1]), rainfall, color=cmap[0])
    #
    # ax.grid('on')
    # ax.set_title(f'Class Activation Map, Class {class_label}, {region.capitalize()}')
    # ax.set_ylabel(f'EVI')
    # ax.set_xticklabels(ticknames, rotation=40)
    # ax.xaxis.set_major_locator(IndexLocator(3, 0))
    # ax.xaxis.set_minor_locator(FixedLocator(minors))
    # ax.tick_params(axis='x', which='both', length=2)
    #
    # ax01.set_ylabel('Rainfall (mm)')
    #
    # count += 1
    # cbar.ax.set_title('Normalized Logit\nImportance')
    # plt.tight_layout()
    #
    # # plt.savefig(f'../figures/class_activation_maps/{region}/'
    # #             f'cam_model_{model_id}_pred_{pred_label}_actual_{orig_label}_plot_{ix}.png')
    # cbar.remove()



if __name__  == '__main__':

    viz_cam()