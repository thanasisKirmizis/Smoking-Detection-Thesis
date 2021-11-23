# Smoking Detection Thesis

## Description
This repository holds scripts and functions used in my thesis titled "A Bottom-up method Towards the Automatic and Objective Monitoring of Smoking Behavior In-the-wild using Wrist-mounted Inertial Sensors".

You can read more about the method/experiments in the published paper: https://arxiv.org/abs/2109.03475

You can also download and load the dataset used for this experiment from here: https://zenodo.org/record/4507451#.YZz_BtAzaiM

## Usage
Clone the whole directory tree, insert your compatible dataset into the "datasets" folder and run `main.py`.

In overview, the steps that are followed in this experiment are:

- Dataset reading, preprocessing and saving into pickle files for later use.
- Model training with the user-defined parameters.
- Results evaluation.

## Technical details

For the training of the network, GPU usage is recommended. If you have an AMD GPU, then you should probably use PLAIDML as Keras backend.
In order to enable that in the code, you will need to uncomment lines 14-16 inside the `perform_training.py` module.
