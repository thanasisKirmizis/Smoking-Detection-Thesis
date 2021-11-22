# Imports
import pickle
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from keras import backend as K
from auxiliary_functions import calc_tp_fp_fn

### ------- PLAIDML-Keras Setup ------- ###
### Uncomment below if using an AMD GPU with PLAIDML installed ###

# import plaidml.keras
# plaidml.keras.install_backend()
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

### ----------------------------------- ###

def evaluate_models(threshold, weight, epochs, augmented, plot_results, drop_gyroscope):
    
    # Keep a list with the confusion matrices for window-to-window evaluation
    conf_mat_list_1 = []
    
    # Keep a list with the confusion matrices for puff-to-puff evaluation
    conf_mat_list_2 = []
    
    # Define directories names
    if drop_gyroscope:
        
        subject_folder = '../subjects/no gyro'
        models_folder = '../models/no gyro'
    else:
        
        subject_folder = '../subjects/with gyro'
        models_folder = '../models/with gyro'
    
    # Read the test data for each subject from pickle files
    for file in os.listdir(subject_folder):
        
        # Skip hidden files (e.g. .gitignore)
        if file.startswith('.'): continue
        
        subj_filename = subject_folder + '/' + file
        
        # Load the dictionary with all needed data for this testing subject
        print("Loading subject data...\n\n")
        with open(subj_filename, 'rb') as f:
            subj_dict = pickle.load(f)
            
        # Load the corresponing trained model for this subject
        print("Loading model ", subj_dict['subj'], "...\n\n")
        model_filename = models_folder + '/aug_epochs_' + str(epochs) + '_model_' + subj_dict['subj'] + '.mdl' if augmented else models_folder + '/raw_epochs_' + str(epochs) + '_model_' + subj_dict['subj'] + '.mdl'
        model = load_model(model_filename)
        
        # Extract the (windowed) testing data with all sessions concatenated
        test_X_W = subj_dict['test_x_w']
        test_Y_W = subj_dict['test_y_w']
        
        # Predict
        print("Predicting...\n\n")
        pred_Y = model.predict(test_X_W)
        
        # Set classes based on probabilities and a set threshold
        for i in range(len(pred_Y)):
            
            if(pred_Y[i] >= threshold): 
                
                pred_Y[i] = 1
            else:
                
                pred_Y[i] = 0
        
        # Find the local maxima between the predicted values
        p, _ = find_peaks(pred_Y[:, 0], distance = 10)
        
        # Plot the comparison of the predictions with the ground truths
        if plot_results:
            
            plt.figure()
            plt.title("Ground Truth Vs Prediction for Subject " + subj_dict['subj'])
            plt.plot(test_Y_W)
            plt.plot(pred_Y)
            plt.plot(p, pred_Y[p, 0], 'x')
        
        # Evaluate
        print("Evaluating with window-to-window approach...\n\n")
        conf_mat_list_1.append(confusion_matrix(test_Y_W, pred_Y, labels = [1, 0]))
        
        print("Evaluating with puff-to-puff approach...\n\n")
        tp,fp,fn = calc_tp_fp_fn(test_Y_W, pred_Y[:, 0], p)
        conf_mat_list_2.append(np.array([[tp, fp], [fn, []]], dtype = object))
        
        print("------------------------------------------\n")
    
    # Display results from window-to-window evaluation
    sum_cms_1 = np.sum(conf_mat_list_1, axis = 0)
    
    w_acc = (sum_cms_1[0, 0] * weight + sum_cms_1[1, 1])/((sum_cms_1[0, 0] + sum_cms_1[1, 0]) * weight + sum_cms_1[1, 1] + sum_cms_1[0, 1])
    rec_1 = sum_cms_1[0, 0] / (sum_cms_1[0, 0] + sum_cms_1[1, 0])
    prec_1 = sum_cms_1[0, 0] / (sum_cms_1[0, 0] + sum_cms_1[0, 1])
    f1_1 = 2 * (rec_1 * prec_1) / (rec_1 + prec_1)
    
    print("Weighted accuracy for window-to-window evaluation is: ", w_acc)
    print("\nF1-score for window-to-window evaluation is: ", f1_1)
    print("\nRecall for window-to-window evaluation is: ", rec_1)
    print("\nPrecision for window-to-window evaluation is: ", prec_1)
    
    # Display results from puff-to-puff evaluation
    sum_cms_2 = np.sum(conf_mat_list_2, axis = 0)
    
    rec_2 = sum_cms_2[0, 0] / (sum_cms_2[0, 0] + sum_cms_2[1, 0])
    prec_2 = sum_cms_2[0, 0] / (sum_cms_2[0, 0] + sum_cms_2[0, 1])
    f1_2 = 2 * (rec_2 * prec_2) / (rec_2 + prec_2)
    
    print("\nF1-score for puff-to-puff evaluation is: ", f1_2)
    print("\nRecall for puff-to-puff evaluation is: ", rec_2)
    print("\nPrecision for puff-to-puff evaluation is: ", prec_2)
