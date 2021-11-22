# Imports
import os
import pickle
import numpy as np
from auxiliary_functions import extract_standard_params, custom_standardize

def save_testing_data(drop_gyroscope):

    # Keep dictionaries that hold the data for each subject
    subjects_X = {}
    subjects_Y = {}
    
    # Define directories names
    if drop_gyroscope:
        
        windows_folder = '../windows/no gyro'
        subject_folder = '../subjects/no gyro'
    else:
        
        windows_folder = '../windows/with gyro'
        subject_folder = '../subjects/with gyro'
    
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
                
    
    # Extract the standardization parameters from the whole training set
    mean_params, std_params = extract_standard_params(subjects_X)
    
    print("\nMeans from the windowed data: ", mean_params, "\n")
    print("\nStDs from the windowed data: ", std_params, "\n")
                
    ### Save the testing data to pickle files ###
    
    # Loop through each subject and ALL of its sessions
    for subj in subjects_X:
    
        print("Saving data of subject ", subj)
        
        # Keep the lists for the testing data
        test_X = []
        test_Y = []
        
        # Fill the test X list
        for sessX in subjects_X[subj]:
            test_X = test_X + sessX
          
        # Fill the test Y list
        for sessY in subjects_Y[subj]:             
            test_Y = test_Y + sessY
            
        # Stack the test list into a 3D numpy array
        test_X = np.stack(test_X, axis = 0)
        
        # Standardize the testing data based on the calculated parameters
        for i in range(len(test_X)):
            test_X[i] = custom_standardize(test_X[i], mean_params, std_params)
        
        # Convert labels of test Y from (-1,1) to (0,1)
        test_Y = np.array(test_Y)
        test_Y[np.where(test_Y == -1)[0]] = 0
        
        # Save to pickle file
        with open(subject_folder + '/subj_' + subj, 'rb') as f:
            subj_dict = pickle.load(f)
        
        subj_dict['test_x_w'] = test_X
        subj_dict['test_y_w'] = test_Y
        
        with open(subject_folder + '/subj_' + subj, 'wb') as f:
              pickle.dump(subj_dict, f)
        
        print("\n------------------------------------------\n")

            

