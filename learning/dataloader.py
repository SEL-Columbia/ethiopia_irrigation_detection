import numpy as np
import pandas as pd
import tensorflow as tf
from glob import glob
import os

class DataGenerator():
    'This selects and prepares training and testing data'
    def __init__(self, args, dir_time, training_regions):

        self.args = args
        self.parent_dir = args.parent_dir
        self.tfrecord_dir = args.tfrecord_dir
        self.train_config = args.train_config
        self.imagery_type = args.imagery_type
        self.training_regions = training_regions
        self.calculate_normalization = args.calculate_normalization
        self.num_epochs = args.num_epochs
        self.train_test_only = args.train_test_only
        self.frac_training = args.frac_training
        self.test_only = args.test_only

        self.base_batch_size = args.base_batch_size
        self.num_images_for_norm = 10000

        self.all_regions = [ 'tana', 'rift', 'koga', 'kobo', 
                             'alamata', 'liben', 'jiga', 'motta']


        self.parent_dir = args.parent_dir
        self.dir_time = dir_time
        self.existing_norm = args.existing_norm

        
        self.n_feats = 360


        self.create_dirs()

        self.load_tfrecord_filenames()

        self.determine_class_balancing_loss()

        self.determine_batch_size()
        #
        self.determine_regional_sample_weighting()
        #
        self.load_datasets()

    def create_dirs(self):

        if not os.path.exists(self.args.normalizations_dir):
            os.makedirs(self.args.normalizations_dir)
        if not os.path.exists(f'{self.args.trained_models_dir}/{self.args.model_type}'):
            os.makedirs(f'{self.args.trained_models_dir}/{self.args.model_type}')


    def load_tfrecord_filenames(self):

        self.tfr_dict = {}
        self.training_px_dict = {}


        if self.train_test_only:
            configs = ['training', 'testing']
        
        if self.test_only:
            configs = ['testing']
        
        if not self.train_test_only and not self.test_only:
            configs = ['training', 'validation', 'testing']


        for ix, region in enumerate(self.all_regions):
            for config in configs:
                dir_name = f'{self.parent_dir}/{self.tfrecord_dir}/{region}/tfrecords'
                #print(dir_name)

                if self.train_test_only:
                    tfr_file = glob(f'{dir_name}/{config}_*{self.frac_training}*.tfrecord')[0]
                    print(f'Loading tfrecord: {tfr_file}')
                else: 
                    tfr_file = glob(f'{dir_name}/*{config}_*.tfrecord')[0]


                #print(tfr_file)
                self.tfr_dict[f'{region}_{config}'] = tfr_file


        for key in self.tfr_dict.keys():
            if 'training' in key:
                fn = self.tfr_dict[key].split('/')[-1]
                irrig_px = int(fn.split('_')[-3])
                noirrig_px = int(fn.split('_')[-1].replace('.tfrecord',''))
                
                num_training_pixels = irrig_px + noirrig_px
                
                # Add num irrig px, num no-irrig px, total px
                self.training_px_dict[key] = [irrig_px, noirrig_px, num_training_pixels]

                # print(f'Key in dictionary: {key}')
                # print(f'Value: {self.training_px_dict[key]}')

        if self.test_only:
            for ix, region in enumerate(self.all_regions):
                self.training_px_dict[f'{region}_training'] = [1, 1, 1]

    def determine_class_balancing_loss(self):

        self.loss_dict = {}

        for region in self.all_regions:
            training_px = self.training_px_dict[f'{region}_training']

            self.loss_dict[f'{region}_class_weights'] = training_px[2] / (
                    2 * np.array([training_px[0], training_px[1]]))

        print('Loss dictionary')
        print([(i, v) for i, v in self.loss_dict.items()])

    def determine_regional_sample_weighting(self):

        self.sample_weighting_dict = {}

        batch_sizes = [val[2] for (i, val) in self.training_px_dict.items()]
        # print(batch_sizes)
        max_batch_size = np.max(batch_sizes)

        for region in self.all_regions:
            self.sample_weighting_dict[region] = max_batch_size / self.training_px_dict[f'{region}_training'][2]

        print('Regional weights')
        print([(i, v) for i, v in self.sample_weighting_dict.items()])

    def determine_batch_size(self):

        self.batch_size_dict = {}
        self.num_batches_dict = {}

        total_training_px = []

        for key in self.training_px_dict.keys():
            total_training_px.append(self.training_px_dict[key][2])

        min_px = np.min(total_training_px)

        for region in self.all_regions:

            region_train_pixels = self.training_px_dict[f'{region}_training'][2]
            self.batch_size_dict[region] = int(self.base_batch_size * region_train_pixels/min_px)
            self.num_batches_dict[region] = region_train_pixels/self.batch_size_dict[region]

        self.num_batches_dict['per_epoch'] = int(np.min([self.num_batches_dict[key] for key
                                                     in self.num_batches_dict.keys()]))

        print('Num batches dictionary')
        print([(i,v) for i,v in self.num_batches_dict.items()])

        print('Batch size dictionary')
        print([(i,v) for i,v in self.batch_size_dict.items()])

    def parse_example(self, example_proto):

        features = {
                "features": tf.io.FixedLenSequenceFeature(
                    [], dtype=tf.float32, allow_missing=True),
                "label": tf.io.FixedLenFeature([], tf.int64),
            }

        image_features = tf.io.parse_single_example(example_proto, features)

        features = image_features['features']
        labels = image_features['label']

        return features, labels

    def adjust_labels(self, features, labels):

        labels = labels - 1

        return features, labels


    def normalization_func(self, band_means, band_stds, features, labels):


        features = tf.cast(features, tf.float64)
        features = tf.math.divide_no_nan((features - band_means), band_stds)

        return features, labels

    def extract_evi(self, features, labels):


        ts_list = []
        num_bands_per_ts = 10
        
        for ix in range(36):
            ts_vals = features[...,ix*num_bands_per_ts:(ix+1)*num_bands_per_ts]
            ts_list.append(ts_vals)                                                        

        features = tf.cast(tf.stack(ts_list, axis=1), tf.float32)

        # Extract EVI
        features = 2.5*(features[..., 6] - features[..., 2]) / (features[..., 6] + 6*features[..., 2] - 7.5*features[..., 0] + 10000)

        # Limit to [0, 1]
        features = tf.clip_by_value(features, clip_value_min=0, clip_value_max=1)
        
        # Normalize
        features = tf.math.divide((features[..., None] - 0.2555), 0.16886)


        return features, labels

    def extract_evi_evi_ds_only(self, features, labels):
        
        # Cast as float
        features = tf.cast(features, tf.float32)

        # Limit to [0, 1]
        features = tf.clip_by_value(features, clip_value_min=0, clip_value_max=1)

        # Normalize
        features = tf.math.divide((features[..., None] - 0.2555), 0.16886)

        return features, labels


    def shift_timeseries(self, features, labels):
        # input of shape (batch, timesteps, bands)

        max_shift = self.args.max_shift # 30 days in either way
        shifted_features_list = []

        features = tf.unstack(features, axis=0)

        for ts in features:
            shift = np.random.randint(low=-max_shift, high=max_shift+1)
            shifted_features = tf.concat((ts[shift::, ...], ts[0:shift, ...]), axis=0)[None,...]

            shifted_features_list.append(shifted_features)

        features_out = tf.concat(shifted_features_list, axis=0)


        return features_out, labels


    def s2_timeseries_input_fx(self, chirps_ts, features, label):

        chirps_ts = tf.cast(chirps_ts, tf.float32)

        ts_list = []
        num_bands_per_ts = 10
        for ix in range(36):
            ts_vals = tf.cast(features[...,ix*num_bands_per_ts:(ix+1)*num_bands_per_ts], tf.float32)
            chirps_tiled = tf.tile(chirps_ts[ix][None,...], [tf.shape(ts_vals)[0]])[...,None]
            ts_w_chirps = tf.concat([ts_vals, chirps_tiled], axis=-1)

            ts_list.append(ts_w_chirps)

        features = tf.cast(tf.stack(ts_list, axis=1), tf.float32)

        return features, label
    
    def apply_normalizations_and_input_fxs(self):

        if self.calculate_normalization:
            ## Calculate normalization using the DS
            norm_df = pd.DataFrame()

            for region in self.all_regions:
                print(f'Calculating normalization: {region}')
                ds = self.ds_dict[f'training_{region}']

                input_mean_array = np.zeros((int(np.ceil(self.training_px_dict[f'{region}_training'][2]/
                                                         self.batch_size_dict[region])),  self.n_feats))
                input_std_array = np.zeros((int(np.ceil(self.training_px_dict[f'{region}_training'][2] /
                                                        self.batch_size_dict[region])),  self.n_feats))

                for ix, (features, label) in enumerate(ds.take(len(input_mean_array))):
                    input_mean_array[ix] = np.mean(features, axis=0)
                    input_std_array[ix] = np.std(features, axis=0)

                # Calculate means + standards deviations
                input_mean = np.mean(input_mean_array, axis=0)
                input_std = np.mean(input_std_array, axis=0)
                print(f'Feature means for region {region}: {input_mean}')
                print(f'Feature stds for region {region}: {input_std}')

                norm_df[f'{region}_mean'] = input_mean
                norm_df[f'{region}_std'] = input_std


            norm_file = f'../data/normalizations/norm_stats_{self.dir_time}.csv'
            norm_df.to_csv(norm_file)
        else:
            norm_file = f'../data/normalizations/norm_stats_{self.existing_norm}.csv'

        norm_df = pd.read_csv(norm_file, index_col=0)

        for key, ds in self.ds_dict.items():
            if not self.args.evi_only:

                self.norm_df = norm_df

                region = key.split('_')[-1]
                band_means = norm_df[f'{region}_mean']
                band_stds = norm_df[f'{region}_std']

                norm_func = lambda features, label: self.normalization_func(band_means, band_stds, features, label)
                funcs_for_ds = [norm_func]

            
                if self.imagery_type == 's2_timeseries':
                    chirps_dir = f'{self.parent_dir}/chirps/regional_average_ts'
                    chirps_ts = pd.read_csv(f'{chirps_dir}/{region}_chirps_avg_20200601-20210601_mm.csv', index_col=0)
                    chirps_ts = chirps_ts['rainfall_mm'].values

                    ## Standardize CHIRPS TS
                    chirps_ts = (chirps_ts - np.mean(chirps_ts))/np.std(chirps_ts)

                    s2_func = lambda features, labels: self.s2_timeseries_input_fx(chirps_ts, features, labels)
                    funcs_for_ds.append(s2_func)

            else:
                funcs_for_ds = []

                evi_func = lambda features, labels: self.extract_evi_evi_ds_only(features, labels)
                funcs_for_ds.append(evi_func)



            # Add shift by timestep
            if self.args.shift_train_val:
                if 'training' in key or 'validation' in key:
                    funcs_for_ds.append(self.shift_timeseries)


            for func in funcs_for_ds: 
                ds = ds.map(func, num_parallel_calls=tf.data.experimental.AUTOTUNE)

            self.ds_dict[key] = ds.prefetch(1)

    def convert_training_ds_to_iterators(self):
        for key, ds in self.ds_dict.items():
            if 'training' in key:
                self.ds_dict[key] = iter(ds.repeat(int(self.num_epochs*2)))  


    def load_datasets(self):

        self.ds_dict = {}

        if self.train_test_only:
             model_configs= ['training', 'testing']

        if self.test_only:
            model_configs = ['testing']

        if not self.train_test_only and not self.test_only:
            model_configs = ['training', 'validation', 'testing']

        for region in self.all_regions:
            for config in model_configs:
                tfr_path = self.tfr_dict[f'{region}_{config}']
                ds = tf.data.TFRecordDataset(tfr_path).map(
                    self.parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
                )

                process_fxs = [self.adjust_labels]
        

                ## APPLY OTHER FUNCTIONS TO FEATURES HERE
                for fx in process_fxs:
                    ds = ds.map(fx, num_parallel_calls=tf.data.experimental.AUTOTUNE)


                # Assign to the dataset dictionary

                if config == 'training':
                    self.ds_dict[f'{config}_{region}'] = ds.shuffle(10000).batch(self.batch_size_dict[region],
                                                                             drop_remainder=True)
                else:
                    self.ds_dict[f'{config}_{region}'] = ds.batch(self.batch_size_dict[region], drop_remainder=True)

        if not self.args.evi_only:
            self.apply_normalizations_and_input_fxs()

        if self.args.model_type not in ['random_forest', 'catboost', 'threshold']:
            print('convert to iter')
            self.convert_training_ds_to_iterators()
