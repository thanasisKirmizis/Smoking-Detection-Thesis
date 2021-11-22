# Imports
from read_dataset import read_dataset
from save_testing_data import save_testing_data
from perform_training import perform_training
from evaluate_models import evaluate_models

# User defined parameters
DATASET = 'SED.pkl'                 # 'SED.pkl' for Smoking Event Detection dataset
WINDOW_SIZE = 225                   # in samples (225 samples = 4.5s)
WINDOW_STEP = 25                    # in samples (25 samples = 0.5s)
EPSILON = 25                        # in samples (distance from the end of a +1 pulse to count the window as puff. 25 samples = 0.5s)
DROP_GYROSCOPE = False              # whether to drop gyroscope measurements or not
FILTER_GRAVITY = True               # whether to filter out the DC component of the gravity from the accelerometer measurements or not
CUTOFF = 1                          # cutoff frequency in Hz to be used in gravity filtering
EPOCHS = 20                         # number of epochs for the training of the network
IS_AUGMENTED = True                 # whether to artificially augment the dataset or not
THRESHOLD = 0.8                     # threshold in order for a prediction probability to count as positive or not
WEIGHT = 10                         # weight in order to calculate weighted accuracy
PLOT_RESULTS = True                 # whether to plot results or not

# Step 1: Read selected dataset, extract windows and corresponding labels and save them for later use
read_dataset(DATASET, CUTOFF, WINDOW_SIZE, WINDOW_STEP, EPSILON, DROP_GYROSCOPE, FILTER_GRAVITY)

# Step 2: Save testing data for each subject, in order to use it later for evaluation
save_testing_data(DROP_GYROSCOPE)

# Step 3: Perform training using the LOSO technique and save the produced models into a folder
perform_training(EPOCHS, IS_AUGMENTED, WINDOW_SIZE, DROP_GYROSCOPE)

# Step 4: Evaluate trained models on controlled dataset using LOSO
evaluate_models(THRESHOLD, WEIGHT, EPOCHS, IS_AUGMENTED, PLOT_RESULTS, DROP_GYROSCOPE)