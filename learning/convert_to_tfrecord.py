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



def return_total_px_and_irrig_frac(region_2, train_config):
    if train_config == 'full_dataset':
        if region_2 == 'oromia':
            total_px = 95154
            frac_irrig = 0.0347
            irrig_px_frac = 1
            noirrig_px_frac = 1
        elif region_2 == 'tana':
            total_px = 186796
            frac_irrig = 0.2188
            irrig_px_frac = 1
            noirrig_px_frac = 1
        elif region_2 == 'catalonia':
            total_px = 12952255
            frac_irrig = 0.4065
            irrig_px_frac = 1
            noirrig_px_frac = 1
        elif region_2 == 'fresno':
            total_px = 19271902
            frac_irrig = 0.9149
            irrig_px_frac = 1
            noirrig_px_frac = 1

    elif train_config == 'region_balanced':
        if region_2 == 'catalonia':
            total_px = 281950
            frac_irrig = 0.4065
            irrig_px_frac = 0.02176841
            noirrig_px_frac = 0.02176841
        elif region_2 == 'fresno':
            total_px = 281950
            frac_irrig = 0.9149
            irrig_px_frac = 0.014630108
            noirrig_px_frac = 0.014630108
        else:
            print(f'Invalid region ({region_2}) for given train configuration: {train_config}')

    elif train_config == 'class_region_balanced':
        if region_2 == 'catalonia':
            total_px = 281950
            frac_irrig = 0.5
            irrig_px_frac = 0.026773616
            noirrig_px_frac = 0.018339857
        elif region_2 == 'fresno':
            total_px = 281950
            frac_irrig = 0.5
            irrig_px_frac = 0.007995153
            noirrig_px_frac = 0.085994764
        else:
            print(f'Invalid region ({region_2}) for given train configuration: {train_config}')

    elif train_config == 'class_region_balanced_10x':
        if region_2 == 'catalonia':
            total_px = 2819500
            frac_irrig = 0.5
            irrig_px_frac = 0.267736155
            noirrig_px_frac = 0.183398575
        elif region_2 == 'fresno':
            total_px = 2819500
            frac_irrig = 0.5
            irrig_px_frac = 0.079951531
            noirrig_px_frac = 0.859947638
        else:
            print(f'Invalid region ({region_2}) for given train configuration: {train_config}')

    return total_px, frac_irrig, irrig_px_frac, noirrig_px_frac


def make_tfrecord_from_folder(folder_name, imagery_type, region_2, train_config):

    all_csvs = sorted(glob(f'{folder_name}/saved_imagery/{region_2}/labeled_pixels/{imagery_type}/*.csv'))
    print(len(all_csvs))
    np.random.shuffle(all_csvs)

    dir =  f'{folder_name}/training_data/tfrecords'
    train_frac = 0.7
    val_frac = 0.15
    test_frac = 1 - train_frac - val_frac


    # or 'region_balanced', class_region_balanced, class_region_balanced_10x for catalonia or fresno

    total_px, frac_irrig, irrig_px_frac, noirrig_px_frac = return_total_px_and_irrig_frac(region_2, train_config)

    training_out_filename = f'{dir}/training/{region_2}_{imagery_type}_{train_config}_fracirrig_' \
                            f'{frac_irrig}_totalpx_{int(train_frac*total_px)}.tfrecord'
    validation_out_filename = f'{dir}/validation/{region_2}_{imagery_type}_{train_config}_fracirrig_' \
                            f'{frac_irrig}_totalpx_{int(val_frac*total_px)}.tfrecord'
    testing_out_filename = f'{dir}/testing/{region_2}_{imagery_type}_{train_config}_fracirrig_' \
                            f'{frac_irrig}_totalpx_{int(test_frac*total_px)}.tfrecord'

    train_writer = tf.io.TFRecordWriter(training_out_filename)
    val_writer = tf.io.TFRecordWriter(validation_out_filename)
    test_writer = tf.io.TFRecordWriter(testing_out_filename)


    print(f'Fraction of irrigated pixels selected: {irrig_px_frac}')
    print(f'Fraction of non-irrigated pixels selected: {noirrig_px_frac}')

    for csv in tqdm(all_csvs):

        full_data = pd.read_csv(csv, index_col=0)

        irrig_data   = full_data[full_data[full_data.columns[-1]] == 2].values
        noirrig_data = full_data[full_data[full_data.columns[-1]] == 1].values

        # print(f'Num irrigated samples: {len(irrig_data)}')
        # print(f'Num non-irrigated samples: {len(noirrig_data)}')

        ## Shuffle
        np.random.shuffle(irrig_data)
        np.random.shuffle(noirrig_data)

        # Select the correct amount of irrigated/non-irrigated pixels
        irrig_data = irrig_data[0:int(irrig_px_frac * len(irrig_data))]
        noirrig_data = noirrig_data[0:int(noirrig_px_frac * len(noirrig_data))]

        data = np.concatenate((irrig_data, noirrig_data), axis=0)

        # Shuffle again
        np.random.shuffle(data)

        ## Split among training, validation, and testing datasets
        train_data = data[0:int(train_frac*len(data))]
        val_data   = data[int(train_frac*len(data)): int((train_frac+val_frac)*len(data))]
        test_data = data[int((train_frac+val_frac)*len(data))::]

        # print(f'Train data row 1: {train_data[0]}')
        # print(f'Val data row 1: {val_data[0]}')
        # print(f'Test data row 1: {test_data[0]}')

        # print(f'Training data length: {len(train_data)}')
        # print(f'Validation data length: {len(val_data)}')
        # print(f'Testing data length: {len(test_data)}')


        for ix, row in enumerate(train_data):
            example = tf.train.Example()
            features, label = row[:-1], int(row[-1])

            if imagery_type == 'dspc_feats':
                features = features.astype(np.float32)
                example.features.feature["features"].float_list.value.extend(features)
            elif imagery_type == 's2_timeseries':
                features = features.astype(np.uint16)
                example.features.feature["features"].int64_list.value.extend(features)

            example.features.feature["label"].int64_list.value.append(label)
            train_writer.write(example.SerializeToString())

        for ix, row in enumerate(val_data):
            example = tf.train.Example()
            features, label = row[:-1], int(row[-1])

            if imagery_type == 'dspc_feats':
                features = features.astype(np.float32)
                example.features.feature["features"].float_list.value.extend(features)
            elif imagery_type == 's2_timeseries':
                features = features.astype(np.uint16)
                example.features.feature["features"].int64_list.value.extend(features)

            example.features.feature["label"].int64_list.value.append(label)
            val_writer.write(example.SerializeToString())

        for ix, row in enumerate(test_data):
            example = tf.train.Example()
            features, label = row[:-1], int(row[-1])

            if imagery_type == 'dspc_feats':
                features = features.astype(np.float32)
                example.features.feature["features"].float_list.value.extend(features)
            elif imagery_type == 's2_timeseries':
                features = features.astype(np.uint16)
                example.features.feature["features"].int64_list.value.extend(features)

            example.features.feature["label"].int64_list.value.append(label)
            test_writer.write(example.SerializeToString())

def make_csv_tfrecord_from_folder(region):

    data_dict = {}

    data_dict['tana'] = [[102284, 27145], [22850, 6386], [20792, 7339]]
    data_dict['rift'] = [[92157, 131942], [19149, 20584], [20378, 25787]]
    data_dict['koga'] = [[150378, 126327], [29661, 28637], [27953, 31423]]
    data_dict['kobo'] = [[93838, 140709], [30549, 41620], [31473, 52917]]
    data_dict['fresno'] = [[127860, 133196], [29326, 27220], [33775, 31725]]
    data_dict['catalonia'] = [[161860, 122495], [40365, 23680], [34523, 24193]]


    configs = ['training', 'validation', 'testing']

    main_dir = f'/Volumes/sel_external/ethiopia_irrigation/{region}/training_data'

    for ix, config in enumerate(configs):
        data = data_dict[region][ix]

        # writer_file = f'{main_dir}/tfrecords/{config}_irrigpx_{data[1]}_noirrigpx_{data[0]}.tfrecord'
        # writer = tf.io.TFRecordWriter(writer_file)

        csv_files = [i for i in glob(f'{main_dir}/csvs/*.csv') if f'_{config}_' in i]

        for jx, csv in enumerate(tqdm(csv_files)):
            full_data = pd.read_csv(csv, index_col=0).values

            for row in full_data:
                features, label = row[0:-1], int(row[-1])

                example = tf.train.Example()

                features = features.astype(np.uint16)



                example.features.feature["features"].int64_list.value.extend(features)
                example.features.feature["label"].int64_list.value.append(label)
                # writer.write(example.SerializeToString())


def make_flat_tfrecord_from_tif_folder(region1, region2):
    np.random.seed(7)

    data_dict = {}
    data_dict['oromia'] = [[2351, 64431, 830], [306, 13971, 394], [644, 13451, 359]]
    data_dict['tana'] = [[27145, 102284, 809], [6386, 22850, 330], [7339, 20792, 323]]
    data_dict['fincha'] = [[435705, 496670, 439], [103088, 113466, 94], [90173, 93208, 95]]
    data_dict['dabat'] = [[204287, 293284, 275], [57204, 49523, 59], [46370, 40132, 60]]
    data_dict['kobo'] = [[467678, 446067, 329], [106663, 102914, 71], [85797, 80513, 72]]
    data_dict['rift'] = [[388474, 458144, 279], [89063, 85528, 60], [83407, 113375, 60]]
    data_dict['catalonia'] = [[329104, 502428, 363], [61226, 100142, 78], [102743, 92249, 78]]
    data_dict['fresno'] = [[357944, 278073, 330], [79780, 61994, 70], [77618, 75549, 72]]

    configs = ['training', 'validation', 'testing']

    tif_dir = f'/Volumes/sel_external/ethiopia_survey/{region1}/saved_imagery/{region2}/training_splits/s2_timeseries'

    for ix, config in enumerate(configs):
        data = data_dict[region2][ix]

        writer_file = f'{tif_dir}/tfrecords/{config}_flatlabelsonly_irrigpx_{data[0]}_' \
                      f'noirrigpx_{data[1]}_ntifs_{data[2]}.tfrecord'
        writer = tf.io.TFRecordWriter(writer_file)

        tif_files = glob(f'{tif_dir}/{config}/*.tif')
        np.random.shuffle(tif_files)

        valid_noirr = 0
        valid_irr = 0
        invalid = 0

        for jx, tif in enumerate(tqdm(tif_files)):
            with rasterio.open(tif, 'r') as src:

                orig_img = src.read()

                img_data = np.transpose(orig_img[0:-1], (1, 2, 0))
                target_data = np.transpose(orig_img[-1])

                valid_pixels = np.where((target_data == 1) | (target_data == 2))

                valid_img = img_data[valid_pixels]
                valid_tar = target_data[valid_pixels]

                if jx == 0:
                    print(f'img_data shape: {img_data.shape}')
                    print(f'target_data shape: {target_data.shape}')

                    print(f'valid img_data shape: {valid_img.shape}')
                    print(f'valid target_data shape: {valid_tar.shape}')

                for kx in range(len(valid_img)):
                    features = valid_img[kx]
                    label = int(valid_tar[kx])

                    if label == 1:
                        valid_noirr +=1
                    elif label == 2:
                        valid_irr +=1
                    else:
                        invalid += 1
                        print(label)


                    # if kx == 0:
                    #     print(features.shape)
                    #     print(label)

                    example = tf.train.Example()

                    features = features.astype(np.uint16)
                    example.features.feature["features"].int64_list.value.extend(features)
                    example.features.feature["label"].int64_list.value.append(label)
                    writer.write(example.SerializeToString())



if __name__ == '__main__':

    regions = ['rift', 'koga', 'kobo', 'fresno', 'catalonia']

    for ix, region in enumerate(regions):
        make_csv_tfrecord_from_folder(region)