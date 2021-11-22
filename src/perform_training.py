# Imports
import pickle
import numpy as np
import os
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, LSTM, Dropout
from keras import backend as K
from auxiliary_functions import extract_standard_params, custom_standardize, augment_dataset

### ------- PLAIDML-Keras Setup ------- ###
### Uncomment below if using an AMD GPU with PLAIDML installed ###

# import plaidml.keras
# plaidml.keras.install_backend()
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

### ----------------------------------- ###

def perform_training(epochs, augmented, window_size, drop_gyroscope):
    
    # Keep dictionaries that hold the data for each subject
    subjects_X = {}
    subjects_Y = {}
    
    # Define directories names
    if drop_gyroscope:
        
        windows_folder = '../windows/no gyro'
        models_folder = '../models/no gyro'
    else:
        
        windows_folder = '../windows/with gyro'
        models_folder = '../models/with gyro'
    
    # Read the data of each subject from pickle files
    for file in os.listdir(windows_folder):
        
        # Skip hidden files (e.g. .gitignore)
        if file.startswith('.'): continue
        
        filename = windows_folder + '/' + file
        
        if 'labels' in file:
            
            subj = file[file.find('subj_') + 5 : file.find('_sess')]
            sess = file[file.find('sess_') + 5 : ]
            
            with open(filename, 'rb') as f:
                windows_labels = pickle.load(f)
             
            if(sess == '1'):
                
                subjects_Y[subj] = [windows_labels]
            else:
                subjects_Y[subj].append(windows_labels)
            
                
        else:
            
            subj = file[file.find('subj_') + 5 : file.find('_sess')]
            sess = file[file.find('sess_') + 5 : ]
    
            with open(filename, 'rb') as f:
                windows = pickle.load(f)
             
            if(sess == '1'):
                
                subjects_X[subj] = [windows]
            else:
                subjects_X[subj].append(windows)
                
    # Artifically augment the dataset
    if augmented:
        subjects_X, subjects_Y = augment_dataset(subjects_X, subjects_Y, drop_gyroscope)
    
    # Extract the standardization parameters from the whole training set
    mean_params, std_params = extract_standard_params(subjects_X)
    
    print("\nMeans from the windowed data: ", mean_params)
    print("\nStDs from the windowed data: ", std_params, "\n")
                
    ### Perform the training by following the LOSO method ###
    
    # Loop through each subject and all of its sessions
    for subj in subjects_X:
    
        print("Leaving out subject ", subj)
        
        ### ------ Training Data ------ ###
        
        # Keep the lists for the training data
        train_X = []
        train_Y = []
        
        # For all the rest subjects other than the one we're leaving out
        for s in subjects_X:
            
            if (s != subj):
                
                # Fill the train X list
                for sessX in subjects_X[s]:
                    train_X = train_X + sessX
                  
                # Fill the train Y list
                for sessY in subjects_Y[s]:                
                    train_Y = train_Y + sessY
        
        # Stack the train list into a 3D numpy array
        train_X = np.stack(train_X, axis = 0)
        
        # Standardize the training data based on the calculated parameters
        for i in range(len(train_X)):
            train_X[i] = custom_standardize(train_X[i], mean_params, std_params)
            
        # Convert labels of train Y from (-1,1) to (0,1)
        train_Y = np.array(train_Y)
        train_Y[np.where(train_Y == -1)[0]] = 0
        
        ### ------ Model ------ ###
            
        # Define input shape of the network
        in_shape = (window_size, 3) if drop_gyroscope else (window_size, 6)
        
        # Create a sequential-layered model
        model = Sequential()
        
        # Add model layers
        model.add(Conv1D(32, kernel_size = 5, activation = 'relu', input_shape = in_shape, padding = 'same'))
        model.add(MaxPooling1D(pool_size = 2))
        model.add(Conv1D(64, kernel_size = 3, activation = 'relu', padding = 'same'))
        model.add(MaxPooling1D(pool_size = 2))
        model.add(Conv1D(128, kernel_size = 3, activation = 'relu', padding = 'same'))
        model.add(LSTM(128))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation = 'sigmoid'))
        
        # Compile model using accuracy to measure model performance
        model.compile(optimizer = 'RMSProp', loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        # Train the model using balanced class weights
        weights = compute_class_weight('balanced', classes = [0, 1], y = train_Y)
        history = model.fit(train_X, train_Y, epochs = epochs, class_weight = dict(enumerate(weights)))
            
        # Save the trained model
        model_filename = models_folder + '/aug_epochs_' + str(epochs) + '_model_' + subj + '.mdl' if augmented else models_folder + '/raw_epochs_' + str(epochs) + '_model_' + subj + '.mdl'
        model.save(model_filename)
        
        print("\n------------------------------------------\n")

            
