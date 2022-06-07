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

## Repository Description

`environment.yml`: This file specifies the Python packages required to run the code contained in this repository. Users can create the necessary Pyhton environment via `conda env create -f environment.yml`.

The `data_and_outputs/` folder is too large to include in this repository. It is instead made available [here](https://console.cloud.google.com/storage/browser/terry_phd_export/projects/ethiopia/ethiopia_irrigation_detection). This folder contains training data, predictions, administrative shapefiles, and phenology maps, all used for in the associated [publication](https://www.frontiersin.org/articles/10.3389/frsen.2022.871942/full). A readme contained within the folder further describes the files saved within. 

The `learning/` folder contains all files necessary to train the machine learning-based classification models for irrigation detection. The scripts are all commented throughout and are described below:

`learning/convert_to_tfrecord.py`: This file converts training data saved as `.csvs` to `.tfrecords`, as Tensorflow can process these files more efficiently, leading to faster training. 

* `learning/dataloader.py`: This file loads in training/validation/testing `.tfrecords` into `tf.data.Dataset` objects. It also applied specified preprocessing functions. 

* `learning/loss.py`: This file contains the binary crossentropy loss function applied during training. 

* `learning/model.py`: This file specifies the three different types of neural network-based models that are tested for prediction performance: A baseline neural network, comprised of 1D convolutions; An LSTM-based network; and a Tranformer-based network. 

* `learning/params.yaml`: This file contains user-specified parameters for model training.

* `learning/training_main_iterable_nn.py`: This is the main script for training neural network (NN) based irrigation detection models -- the baseline NN model, the LSTM-based model, and the Transformer based model. This is the script to run to train one of these types of models, or to load a pretrained NN model for inference. 

* `learning/training_main_iterable_tree.py`: This is the main script for training tree based irrigation detection models -- Random Forest and CatBoost models. Run this script to train a new or to load a pretrained tree-based model. 

The `utility_scripts/` folder contains a series of scripts for interpreting and processing irrigation detection model results. These include: 

* `utility_scripts/cam_visualization.py`: This script allows users to visualize a class activation map (CAM) for the neural network based irrigation detectors. The CAM reveals which timeseteps within the input timeseries are highly important for accuracte predictions. For futher reference, the class activation mapping strategy was first introduced in the following [paper](https://arxiv.org/pdf/1512.04150.pdf). 

* `utility_scripts/cluster_clean_samples.py`: This script performs cluster cleaning of labeled irrigated and non-irrigated samples, per the Methodology described in this repository's associated paper.

* `utility_scripts/histograms_of_irrigated_plot_sizes.py`: This script creates a histogram of irrigated plot sizes, as is seen in Figure VI-1 of the following [PhD disseration](https://storage.googleapis.com/terry_phd_export/thesis/tconlon_phd_dissertation.pdf). 

* `utility_scripts/ks_statistics_calculations.py`: This script calculates the Kolmogorov-Smirnov statistic between regional distributions of irrigated or non-irrigated samples.  

* `utility_scripts/plot_asasa_chirps_evi_ts.py`: This script plots enhanced vegetation index (EVI) timeseries and [Climate Hazards Group InfraRed Precipitation with Station Data](https://www.chc.ucsb.edu/data/chirps) precipitation esimates for an area near Asasa, Ethiopia. 

* `utility_scripts/plot_feature_importances.py`: This script plots the feature importances associated with trained tree-based classification models. 

* `utility_scripts/plot_tEMs.py`: This script plots temporal endmembers (tEMS).

* `utility_scripts/plotting_performance_vs_vt.py`: This script plots classifier model performance against the number of unique regions containing labeled data that are included in training.  

* `utility_scripts/spectral_unmixing.py`: This script contains code for applying a spectral unmixing model to determine the contribution of certain temproal endmembers to vegetation phenologies. See [Small (2012)](https://www.sciencedirect.com/science/article/pii/S0034425712002349) for a full description of the process of using endmember-based unmixing approaches for spatiotemporal vegetation characterization. 
