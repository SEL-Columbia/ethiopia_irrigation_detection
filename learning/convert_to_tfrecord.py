import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import random
from glob import glob
import rasterio


def _uint16_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if isinstance(value, np.ndarray):
        value = value.astype(np.uint16).flatten().tolist()
    elif not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float64_feature(value):
    """Wrapper for inserting float64 features into Example proto."""
    if isinstance(value, np.ndarray):
        value = value.flatten().tolist()
    elif not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_example_tif(img_data, target_data, img_shape, target_shape):
    """ Converts image and target data into TFRecords example.

    Parameters
    ----------
    img_data: ndarray
        Image data
    target_data: ndarray
        Target data
    img_shape: tuple
        Shape of the image data (h, w, c)
    target_shape: tuple
        Shape of the target data (h, w, c)
    dltile: str
        DLTile key

    Returns
    -------
    Example: TFRecords example
        TFRecords example
    """
    if len(target_shape) == 2:
        target_shape = (*target_shape, 1)

    features = {
        "image/image_data": _uint16_feature(img_data),
        "image/height": _uint16_feature(img_shape[0]),
        "image/width": _uint16_feature(img_shape[1]),
        "image/channels": _uint16_feature(img_shape[2]),
        "target/target_data": _uint16_feature(target_data),
        "target/height": _uint16_feature(target_shape[0]),
        "target/width": _uint16_feature(target_shape[1]),
        "target/channels": _uint16_feature(target_shape[2]),
    }

    return tf.train.Example(features=tf.train.Features(feature=features))



def make_tfrecords_from_csvs(region):

    configs = ['training', 'validation', 'testing']

    main_dir = f'/Volumes/sel_external/ethiopia_irrigation_share/training_data/{region}'

    for ix, config in enumerate(configs):

        csv_file = [i for i in glob(f'{main_dir}/csvs/*.csv') if f'_{config}_' in i][0]
        full_data = pd.read_csv(csv_file, index_col=0).values

        num_irrig = np.count_nonzero(full_data[:, -1] == 2)
        num_noirrig = np.count_nonzero(full_data[:, -1] == 1)


        writer_file = f'{main_dir}/tfrecords/{region}_{config}_evi_only_irrigpx_{num_irrig}_noirrigpx_{num_noirrig}.tfrecord'
        writer = tf.io.TFRecordWriter(writer_file)


        for jx, row in enumerate(full_data):
            features, label = row[0:-1], int(row[-1])

            example = tf.train.Example()

            features = features.astype(np.float32)

            example.features.feature["features"].float_list.value.extend(features)
            example.features.feature["label"].int64_list.value.append(label)

            writer.write(example.SerializeToString())



if __name__ == '__main__':

    regions = ['alamata', 'jiga', 'kobo', 'koga',
               'liben', 'motta', 'rift', 'tana']

    for ix, region in enumerate(regions):
        make_tfrecords_from_csvs(region)