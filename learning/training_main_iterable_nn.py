import tensorflow as tf
import numpy as np
import copy
from tqdm import tqdm
import os
from learning.dataloader import DataGenerator
from learning.model import baseline_model, lstm_model, transformer_model
import argparse, yaml
import datetime
import pandas as pd
import itertools

import random as python_random
tf.random.set_seed(42)
np.random.seed(42)
python_random.seed(42)

print(tf)
print(tf.__version__)

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



def calculate_metrics(predictions, true_labels, threshold):
    # Calculate loss for summary writing
    true_labels = tf.cast(true_labels, tf.float32)
    
    # print(predictions[0:20])
    predictions = tf.cast(tf.math.greater(predictions[:, 1], threshold), tf.float32)

    # Round predictions
    predictions = tf.math.round(predictions)

    TP = tf.math.count_nonzero(predictions * true_labels)
    TN = tf.math.count_nonzero((predictions - 1) * (true_labels - 1))
    FP = tf.math.count_nonzero(predictions * (true_labels - 1))
    FN = tf.math.count_nonzero((predictions - 1) * true_labels)

    acc = tf.math.divide((TP + TN), (TP + TN + FP + FN))
    precision = tf.math.divide(TP, (TP + FP))
    recall = tf.math.divide(TP, (TP + FN))
    f1 = tf.math.divide(2 * precision * recall, (precision + recall))


    return TP.numpy(), TN.numpy(), FP.numpy(), FN.numpy()


def get_train_fx():
    @tf.function(experimental_relax_shapes=True)
    def train_step_fx(model, features, labels, irrig_lm, noirrig_lm, region_weight,
                loss_obj, optimizer):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).


            predictions = model(features, training=True)
            #print(predictions)
            labels_onehot = tf.expand_dims(labels, axis=-1)
            labels_onehot = tf.concat([1-labels_onehot, labels_onehot], axis=-1)


            weights = tf.cast(irrig_lm, tf.float32) * tf.cast(labels, tf.float32) + \
                      tf.cast(noirrig_lm, tf.float32)*(1-tf.cast(labels, tf.float32)) #[tf.newaxis, ...]


            loss = loss_obj(tf.cast(labels_onehot, tf.float32), tf.cast(predictions, tf.float32),
                            sample_weight=weights)
    
            loss = loss * tf.cast(region_weight, tf.float32)


            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))


    return train_step_fx



def model_evaluation(model, val_ds, region, thresholds):

    preds_list  = []
    true_labels_list = []

    for ix, (features, true_labels) in val_ds.enumerate():
        predictions = model(features, training=False)
        preds_list.extend(predictions.numpy())
        true_labels_list.extend(true_labels.numpy())

    f1_list = []
    preds_tuple_list = []

    for threshold in thresholds:
        TP, TN, FP, FN = calculate_metrics(np.array(preds_list), np.array(true_labels_list), threshold)
        f1_score = TP / (TP + 0.5*(FP + FN))

        f1_list.append(f1_score)
        preds_tuple_list.append((TP, TN, FP, FN))


    return f1_list, preds_tuple_list, preds_list, true_labels_list

def update_model_weights(args, model, current_results_dict, best_results_dict, v_regions,
                         epoch, best_epoch, checkpoint):



    # Retrieve prior best and current accuracies
    best_scores = []
    current_scores = []

    metric = args.weight_update_criteria

    for v_region in v_regions: 
        #if v_region in ['tana', 'oromia']:
        if metric == 'acc':
            best_scores.extend([best_results_dict[f'{v_region}_irrig'], best_results_dict[f'{v_region}_noirrig']])
            current_scores.extend([current_results_dict[f'{v_region}_irrig'], current_results_dict[f'{v_region}_noirrig']])
        if metric == 'f1':
            best_scores.extend([best_results_dict[f'{v_region}_f1']])
            current_scores.extend([current_results_dict[f'{v_region}_f1']])

    current_score = np.min(current_scores)
    prior_score = np.min(best_scores)

    print(f'Current min {metric}: {current_score}')
    print(f'Prior best min {metric}: {prior_score}')

    if current_score > prior_score:
        print(f'Min F1 score has improved from {prior_score} to {current_score}, checkpointing model')
        best_results_dict = copy.deepcopy(current_results_dict)
        best_epoch = epoch

        checkpoint_prefix = f'{args.CHECKPOINT_DIR}/{args.model_type}/{args.DIR_TIME}/'
        
        manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_prefix, 
                        max_to_keep=1, 
                        checkpoint_name = f'epoch_{epoch}_top_min_f1_{current_score:.4f}')

        manager.save()

    else:
        print(f'Overall validation accuracy has not improved from {prior_score}, not saving new best weights')


    return model, best_results_dict, best_epoch


def return_model(args, model_name):

    input_shape = (args.NUM_TIMESTEPS, args.INPUT_CHANNELS)

    if model_name == 'lstm':
        
        model = lstm_model(
                input_shape,
                n_lstm_nodes=64, 
                n_lstm_layers=4,
                mlp_units=[32], 
                dropout=0.25, 
                mlp_dropout=0.4
                )
    
    elif model_name == 'transformer':
        model = transformer_model(
                input_shape,
                head_size=64,
                num_heads=4,
                ff_dim=4,
                num_transformer_blocks=4,
                mlp_units=[32],
                mlp_dropout=0.4,
                dropout=0.25,
                )

    elif model_name == 'baseline':
        dropout = 0.25
        filters = 64

        model = baseline_model(input_shape, filters=filters, dropout=dropout,
                                mlp_units=[32], mlp_dropout=0.4)

    else:
        print(f'Model name {model_name} not recognized')
        model = None


    return model


def training_function(train_val_regions):

    print('STARTING TRAINING')

    args = get_args()
    args = dotdict(vars(args))

    train_regions = train_val_regions # ['alamata', 'koga', 'kobo', 'liben', 'jiga', 'motta'] #, 'synthetic']
    val_regions   = train_val_regions # ['alamata', 'koga', 'kobo', 'liben', 'jiga', 'motta'] #, 'synthetic']
    
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

    lr = 1e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)


    # Model instantiation
    model_name = args.model_type
    model = return_model(args, model_name)
    train_step = get_train_fx()
    
    if not args.LOAD_EXISTING:
        print('Training from scratch')

        # Define checkpoint object + save path
        checkpoint = tf.train.Checkpoint(
            optimizer=optimizer,
            model=model,
        )

    else:
        print('Loading from existing')

        prev_checkpoint_prefix = f'{args.CHECKPOINT_DIR}/{args.model_type}/{args.pretrained_model}/'
        checkpoint = tf.train.Checkpoint(
            optimizer=optimizer,
            model=model,
        )
        print(f'Loading existing checkpoint: {tf.train.latest_checkpoint(prev_checkpoint_prefix)}')
        status = checkpoint.restore(tf.train.latest_checkpoint(prev_checkpoint_prefix))
        print(status.assert_existing_objects_matched())
    
    
    # Define training loss objects
    if args.log:
        log_dir = "../data/tensorboard_logs/" + dir_time
        summary_writer = tf.summary.create_file_writer(log_dir)

    best_epoch = 0
    loss_obj = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0)

    if args.train:
        print('Training')

        for epoch in range(args.num_epochs):
            np.random.shuffle(train_regions)
            print(f'Epoch: {epoch}')
            num_batches = np.min((generator.num_batches_dict['per_epoch'], 200))
            for batch in tqdm(range(num_batches)):

                for region in train_regions:

                    features, labels = generator.ds_dict[f'training_{region}'].next()
                    #if np.count_nonzero(labels == 0) + np.count_nonzero(labels==1) != len(labels):
                    #    print(f'Incorrect label in target: {labels}')

                    irrig_lm =   generator.loss_dict[f'{region}_class_weights'][0]
                    noirrig_lm =  generator.loss_dict[f'{region}_class_weights'][1]

                    region_weight = tf.cast(generator.sample_weighting_dict[region], tf.float32)
                    #if batch == 0:
                    #    print(f'Region {region}, num features: {features.shape}')
                    #    print(f'Region {region}, irrig loss multiplier: {irrig_lm}')
                    #    print(f'Region {region}, noirrig loss multiplier: {noirrig_lm}')
                    #    print(f'Region {region}, features: {features}')
                    #    print(f'Region {region}, labels: {labels}')



                    train_step(model, features, labels, irrig_lm, noirrig_lm, region_weight,
                            loss_obj, optimizer)

            # Reset class-region loss multiplier for update based on validation performance

            #all_regions = np.unique(val_regions + test_regions)


            for region in val_regions: # + [i for i in test_regions if i not in val_regions]:

                print('------------')
                print(f'Validation set pixels, region: {region}')
                ds = generator.ds_dict[f'validation_{region}']

                thresholds = args.thresholds
                f1_list, preds_tuple_list, preds_list, true_labels_list = model_evaluation(model,
                                                                                           ds,
                                                                                           region=region,
                                                                                           thresholds=thresholds)
                best_ix = np.argmax(f1_list)
                f1_score = f1_list[best_ix]
                (TP, TN, FP, FN) = preds_tuple_list[best_ix]

                print(f'Region {region} validation set f1 score: {f1_score}')
                print(f'Region {region} validation set irrigation accuracy: {TP/(TP+FN)}')
                print(f'Region {region} validation set non-irrigation accuracy: {TN/(TN+FP)}')
                print(f'Region {region} validation set, best threshold: {thresholds[best_ix]}')

                print(f'TP: {TP}')
                print(f'TN: {TN}')
                print(f'FP: {FP}')
                print(f'FN: {FN}')

                # Only add to results_dict if region in val_region
                if region in val_regions:
                    current_results_dict[f'{region}_irrig_acc'] = TP/(TP+FN)
                    current_results_dict[f'{region}_noirrig_acc'] = TN/(TN+FP)
                    current_results_dict[f'{region}_f1'] = f1_score

            model, best_results_dict, best_epoch = update_model_weights(args, model,
                                                                        current_results_dict,
                                                                        best_results_dict,
                                                                        val_regions,
                                                                        epoch,
                                                                        best_epoch,
                                                                        checkpoint)
            if epoch > best_epoch + 10:
                print('Min F1 score has not improved in 10 epochs, stopping training')
                break
                

    if args.test:
        # Set up results df        
        test_df = pd.DataFrame()

        if not args.train:
            dir_time = args.pretrained_model


        # Load best model for testing
        #best_model = lstm_functional(input_shape, apply_gap)
        best_model = return_model(args, model_name)

        optimizer= tf.keras.optimizers.Adam(learning_rate=lr)

        prev_checkpoint_prefix = f'{args.CHECKPOINT_DIR}/{model_name}/{dir_time}/'
        checkpoint = tf.train.Checkpoint(
            optimizer=optimizer,
            model=best_model,
        )

        print(f'Loading existing checkpoint: {tf.train.latest_checkpoint(prev_checkpoint_prefix)}')
        

        status = checkpoint.restore(tf.train.latest_checkpoint(prev_checkpoint_prefix)).expect_partial()

        test_df['model_dir'] = [dir_time]
        best_epoch = tf.train.latest_checkpoint(prev_checkpoint_prefix).split('/epoch_')[-1].split('_')[0]
        test_df['best_epoch'] = [best_epoch]

        for region in test_regions:

            print('------------')
            print(f'Test set pixels, region: {region}')
            ds = generator.ds_dict[f'testing_{region}']
            print(f'Epoch with best saved weights: {best_epoch}')

            #f1_score, acc, TP, TN, FP, FN = model_evaluation(best_model, ds, region=region)

            thresholds = args.thresholds
            f1_list, preds_tuple_list, preds_list, true_labels_list = model_evaluation(model,
                                                                                       ds,
                                                                                       region=region,
                                                                                       thresholds=thresholds)
            
            best_ix = np.argmax(f1_list)
            f1_score = f1_list[best_ix]
            (TP, TN, FP, FN) = preds_tuple_list[best_ix]

            print(f'Region {region} test set f1 score: {f1_score}')
            print(f'Region {region} test set irrigation accuracy: {TP/(TP+FN)}')
            print(f'Region {region} test set non-irrigation accuracy: {TN/(TN+FP)}')
            print(f'Region {region} test set TP: {TP}')
            print(f'Region {region} test set TN: {TN}')
            print(f'Region {region} test set FP: {FP}')
            print(f'Region {region} test set FN: {FN}')

            test_df[f'{region}_pos_acc'] = [TP/(TP+FN)]
            test_df[f'{region}_neg_acc'] = [TN/(TN+FP)]
            test_df[f'{region}_f1_score'] = [f1_score]


            if args.save_test_predictions:
                pred_save_dir = f'../data/results/test_set_predictions/{dir_time}'
                if not os.path.exists(pred_save_dir):
                    os.mkdir(pred_save_dir)

                test_preds_df = pd.DataFrame()
                test_preds_df['predictions'] = preds_list
                test_preds_df['true_labels'] = true_labels_list

                outfile = f'{pred_save_dir}/{region}_testset_predictions_model_{dir_time}.csv'
                test_preds_df.to_csv(outfile)

        
        out_dir = f'../data/results/batch_results/{args.model_type}'
        test_df.to_csv(f'{out_dir}/testing_results_{dir_time}_{"-".join(train_val_regions)}.csv')


if __name__ == '__main__':


    all_training_regions = ['rift', 'alamata', 'koga', 'kobo', 'liben', 'jiga', 'motta']
    
    for i in range(7,0,-1):
        training_regions_list = list(itertools.combinations(all_training_regions, i))

        for regions in tqdm(training_regions_list):
            training_function(list(regions))


