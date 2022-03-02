import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
from learning.dataloader import DataGenerator
import argparse, yaml
import datetime
from random import shuffle
import pandas as pd
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from catboost import CatBoostClassifier

import random as python_random
import joblib
tf.random.set_seed(42)
np.random.seed(42)
python_random.seed(42)

print(f'Tensorflow verion: {tf.__version__}')
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_args():
    parser = argparse.ArgumentParser(
        description="Irrigation detection"
    )

    parser.add_argument(
        "--training_params_filename",
        type=str,
        default="params.yaml",
        help="Filename defining model configuration",
    )

    args = parser.parse_args()
    config = yaml.load(open(args.training_params_filename))
    for k, v in config.items():
        args.__dict__[k] = v

    return args



def return_model(model_name):

    if model_name == 'random_forest':
        model = RandomForestClassifier(n_estimators=1000, verbose=True)

    elif model_name == 'catboost':
        model = CatBoostClassifier(iterations=1000,
                           task_type="GPU",
                           devices='0:1',
                           verbose=False,
                           )

    else:
        print(f'Model name {model_name} not recognized.')
        model = None

    return model



def return_weighted_arrays(args, generator, regions, model_config):

    if model_config == 'training':
        num_batches = np.min((generator.num_batches_dict['per_epoch'], 400)) # 400 usually

    if args.model_type == 'random_forest' or args.model_type == 'catboost':
        training_array = np.empty(shape=(0, args.INPUT_CHANNELS * args.NUM_TIMESTEPS))
    elif args.model_type == 'threshold':
        training_array = np.empty(shape=(0, args.NUM_TIMESTEPS))

    labels_array = np.empty(shape=(0,))
    weights_array = np.empty(shape=(0,))

    print(f'Loading arrays for: {regions}')

    for region in regions:

        irrig_lm = generator.loss_dict[f'{region}_class_weights'][0]
        noirrig_lm = generator.loss_dict[f'{region}_class_weights'][1]
        regional_weight = generator.sample_weighting_dict[region]

        ds = generator.ds_dict[f'{model_config}_{region}']

        if args.model_type == 'random_forest' or args.model_type == 'catboost':
            regional_feats_array = np.empty(shape=(0, args.INPUT_CHANNELS * args.NUM_TIMESTEPS))
        elif args.model_type == 'threshold':
            regional_feats_array = np.empty(shape=(0, args.NUM_TIMESTEPS))

        regional_labels_array = np.empty(shape=(0,))
        regional_weights_array = np.empty(shape=(0,))

        for ix, (features, labels) in ds.enumerate():
            
            features = np.reshape(features.numpy(), (features.shape[0], features.shape[1] * features.shape[2]))
            labels = labels.numpy()
            weights = regional_weight * (irrig_lm * labels + noirrig_lm * -(labels - 1))

            regional_feats_array = np.concatenate((regional_feats_array, features), axis=0)
            regional_labels_array = np.concatenate((regional_labels_array, labels), axis=0)
            regional_weights_array = np.concatenate((regional_weights_array, weights), axis=0)

            if model_config == 'training':
                if ix.numpy() == int(num_batches):
                    break



        training_array = np.concatenate((training_array, regional_feats_array), axis=0)
        labels_array = np.concatenate((labels_array, regional_labels_array), axis=0)
        weights_array = np.concatenate((weights_array, regional_weights_array), axis=0)

    if model_config == 'training':
        training_array, labels_array, weights_array = shuffle(training_array, labels_array, weights_array) 



    return training_array, labels_array, weights_array


def training_function(train_val_regions):

    print('STARTING TRAINING')

    args = get_args()
    args = dotdict(vars(args))

    train_regions = train_val_regions
    val_regions   = train_val_regions
    
    ### NEVER CHANGE THE TEST_REGIONS VARIABLE
    test_regions  = ['tana', 'rift', 'alamata', 'koga', 'kobo', 'liben', 'jiga', 'motta'] #, 'synthetic']

    dir_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args['DIR_TIME'] = dir_time

    generator = DataGenerator(args, dir_time, train_regions)


    best_results_dict = {}
    current_results_dict = {}

    for v_region in val_regions:
        best_results_dict[f'{v_region}_irrig_acc'] = 0
        best_results_dict[f'{v_region}_noirrig_acc'] = 0
        best_results_dict[f'{v_region}_f1'] = 0

        current_results_dict[f'{v_region}_irrig_acc'] = 0
        current_results_dict[f'{v_region}_noirrig_acc'] = 0
        current_results_dict[f'{v_region}_f1'] = 0


    # Model instantiation
    model_name = args.model_type
    model = return_model(model_name)

    if args.train:
        training_array, labels_array, weights_array = return_weighted_arrays(args, generator,
                                                                             train_val_regions, 'training')

        print(f'Training array shape: {training_array.shape}')

        model.fit(X=training_array, y=labels_array, sample_weight=weights_array)


        #print(f'feature_importances: {model.feature_importances_}')

        if args.save_model:
            out_dir = f'../data/trained_models/{args.model_type}'
            out_file = f'{out_dir}/trained_{args.model_type}_{dir_time}_{"-".join(train_val_regions)}'

            if args.model_type == 'random_forest':
                joblib.dump(model, f'{out_file}.joblib')
            elif args.model_type == 'catboost':
                model.save_model(out_file)

    if args.test:
        print('Testing')
        test_df = pd.DataFrame()
        test_df['model_dir'] = [dir_time]


        if args.LOAD_EXISTING:
            model_dir = '../data/trained_models/catboost' 
           
            dir_time = '20211122-163402'
            model_fn = f'{model_dir}/trained_catboost_{dir_time}_tana-rift-alamata-koga-kobo-liben-jiga-motta'

            model = return_model(args, model_name)
            model.load_model(model_fn)

        for region in test_regions:
            print(f'Loading data and predicting over: {region}')
            testing_array, labels_array, weights_array = return_weighted_arrays(args, generator, [region], 'testing')

            print(f'Testing array shape, {region}: {testing_array.shape}')

            pred_labels = model.predict(testing_array)
            conf_matrix = confusion_matrix(labels_array, pred_labels)

            print(f'Conf matrix: {conf_matrix}')
            TN = conf_matrix[0, 0]
            TP = conf_matrix[1, 1]
            FN = conf_matrix[1, 0]
            FP = conf_matrix[0, 1]
            f1_score = TP / (TP + 0.5 * (FP + FN))

            test_df[f'{region}_pos_acc'] = [TP / (TP + FN)]
            test_df[f'{region}_neg_acc'] = [TN / (TN + FP)]
            test_df[f'{region}_f1_score'] = [f1_score]

            if args.save_test_predictions:
                pred_save_dir = f'../data/results/test_set_predictions/{dir_time}'
                if not os.path.exists(pred_save_dir):
                    os.mkdir(pred_save_dir)

                #print(pred_labels[0:5])
                #print(labels_array[0:5])


                pred_save_dir = f'../data/results/test_set_predictions/{dir_time}'
                if not os.path.exists(pred_save_dir):
                    os.mkdir(pred_save_dir)

                test_preds_df = pd.DataFrame()
                test_preds_df['predictions'] = pred_labels
                test_preds_df['true_labels'] = labels_array

                outfile = f'{pred_save_dir}/{region}_testset_predictions_model_{dir_time}.csv'
                test_preds_df.to_csv(outfile)

        out_dir = f'../data/results/batch_results/{args.model_type}'
        test_df.to_csv(f'{out_dir}/testing_results_{dir_time}_{"-".join(train_val_regions)}.csv')


if __name__ == '__main__':


    all_training_regions = ['tana', 'rift'] #, 'alamata', 'koga', 'kobo', 'liben', 'jiga', 'motta']
    
    
    for ix in range(2,0,-1):
        training_regions_list = list(itertools.combinations(all_training_regions, ix))
        for regions in tqdm(training_regions_list):
            training_function(list(regions))


