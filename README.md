# ethiopia_irrigation
Detecting Irrigation in the Ethiopian Highlands.

This repository includes material for a [publication](https://www.frontiersin.org/articles/10.3389/frsen.2022.871942/full) in Frontiers in Remote Sensing, "A Multiscale Spatiotemporal Approach for Smallholder Irrigation Detection".

Corresponding author: Terry Conlon (terence.m.conlon@gmail.com). 

## Overview

This repository contains code and data for detecting irrigation in the Ethiopian Highlands. Through a process detailed in the above linked paper, training data representing a binary distinction -- irrigation vs. no irrigation -- is collected and cleaned. That data is then used to train and test a series of machine-learning based classification models. Further scripts are included that process irrigation predictions and generate insights based on model performance. 

## Repository Structure

```
ethiopia_irrigation_detection
├── environment.yml
├── LICENSE
├── README.md
├── setup.py
├── data_and_outputs/
│   ├── 10m_predictions/
│   ├── administrative_shapefiles/
│   ├── downsampled_predictions/
│   ├── phenology_map/
│   ├── readme.docx
│   ├── thresholded_predictions/
│   ├── training_data/
├── learning/
│   ├── __init__.py
│   ├── convert_to_tfrecord.py
│   ├── dataloader.py
│   ├── loss.py
│   ├── model.py
│   ├── params.yaml
│   ├── training_main_iterable_nn.py
│   ├── training_main_iterable_tree.py
├── utility_scripts/
│   ├── cam_visualization.py
│   ├── cluster_timeseries.py
│   ├── estimating_statistical_distances.py
│   ├── histograms_of_irrigated_plot_sizes.py
│   ├── ks_statistics_calculations.py
│   ├── plot_asasa_chirps_evi_ts.py
│   ├── plot_asasa_s2_evi_ts.py
│   ├── plot_bati_chirps_evi_ts.py
│   ├── plot_rf_feature_importances.py
│   ├── plot_tEMs.py
│   ├── plotting_performance_vs_vt.py
│   ├── spectral_unmixing.py
```

## Root Level Repository Description

`environment.yml`: This file specifies the Python packages required to run the code contained in this repository. Users can create the necessary Pyhton environment via `conda env create -f environment.yml`.

The `data_and_outputs/` folder is too large to include in this repository. It is instead made available [here](). This folder contains training data, predictions, administrative shapefiles, and phenology maps, all used for in the associated [publication](https://www.frontiersin.org/articles/10.3389/frsen.2022.871942/full). A readme contained within the folder further describes the files saved within. 

The `learning/` folder contains all files necessary to train the machine learning-based classification models for irrigation detection. The scripts are all commented throughout and are described below:

`learning/convert_to_tfrecord.py`: This file converts training data saved as `.csvs` to `.tfrecords`, as Tensorflow can process these files more efficiently, leading to faster training. 

`learning/dataloader.py`: This file loads in training/validation/testing `.tfrecords` into `tf.data.Dataset` objects. It also applied specified preprocessing functions. 

`learning/loss.py`: This file contains the binary crossentropy loss function applied during training. 

`learning/model.py`: This file specifies the three different types of neural network-based models that are tested for prediction performance: A baseline neural network, comprised of 1D convolutions; An LSTM-based network; and a Tranformer-based network. 

`learning/params.yaml`: This file contains user-specified parameters for model training.

`learning/training_main_iterable_nn.py`: This is the main script for training neural network (NN) based irrigation detection models -- the baseline NN model, the LSTM-based model, and the Transformer based model. This is the script to run to train one of these types of models, or to load a pretrained NN model for inference. 

`learning/training_main_iterable_tree.py`: This is the main script for training tree based irrigation detection models -- Random Forest and CatBoost models. Run this script to train a new or to load a pretrained tree-based model. 

The `utility_scripts/` folder contains a series of scripts for interpreting and processing irrigation detection model results. These include: 




