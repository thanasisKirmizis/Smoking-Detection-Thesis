# Imports
import numpy as np
import pandas as pd
from scipy.signal import firwin
from scipy.signal import find_peaks

# Functions
def get_windows(df, window_size, window_step):
    
    ''' 
    Extracts the windows of a signal (dataframe) based on the sliding window method
    
        Parameters:
            
            df (pd.DataFrame):                  The dataframe of the signal to extract the windows from
            window_size (int):                  The sliding window length in number of samples
            window_step (int):                  The step of the moving window in number of samples
    
        Returns:
            
            windows (list od pd.DataFrame):     A list of the extracted windows
    '''
    
    windows = []
    
    for i in range(0, len(df) - window_size + 1, window_step):
        
        window = np.array(df.iloc[i : i + window_size, 1: -1])
        windows.append(window)
            
    return windows


def get_window_labels(labels, window_size, window_step, epsilon):
    
    '''
    Extracts the labels (series of +1 and -1) of the windows based on the end-of-window method
    
        Parameters:
            
            labels (list of int):           The signal to extract from in labeled form (with +1 and -1)
            window_size (int):              The sliding window length in number of samples
            window_step (int):              The step of the moving window in number of samples
            epsilon (int):                  The max distance (in samples) from the ending of +1 pulse that a window
                                            can end at and still be labeled as +1		   
        Returns: 
            
            window_labels (list of int):    A list of the extracted window labels
    '''
    
    # Firstly extract the indices of the ending points of the puffs
    ending_idx = [0]
    
    for i in range(len(labels) - 1):
        
        # If located the ending of a puff
        if((labels[i] == 1) and (labels[i + 1] == -1)):
            
            # Keep the ending time of this puff
            ending_idx.append(i)
        
    window_labels = []
    curr_end_idx = 1
	
    for i in range(0, len(labels) - window_size + 1, window_step):
        
        window_end = i + window_size - 1
        
		# If the end of this window falls further than the +1 pulse end + epsilon
        if window_end > ending_idx[curr_end_idx] + epsilon:
			
			# Start comparing with the next +1 pulse
            if curr_end_idx < len(ending_idx) - 1:
                curr_end_idx = curr_end_idx + 1
			
			# Label window as non-puff
            window_labels.append(-1)
		
		# If the end of this window falls closer than the +1 pulse end - epsilon
        elif(window_end < ending_idx[curr_end_idx] - epsilon):
			
			# Label window as non-puff
            window_labels.append(-1)
		
		# If the end of this window falls in between the +1 pulse end +- epsilon
        else:
		
			# Label window as puff
            window_labels.append(1)
             
    return window_labels


def custom_standardize(array, means, stds):
    
    '''
    Standardize a 2D numpy array column-wise based on the inputs mean and std
    
        Parameters:
            
            array (2D ndarray):     a 2D numpy array
            means (1D ndarray):     an 1D array containing the desired means for each column (channel)
            stds (1D ndarray):      an 1D array containing the desired standard deviation for each column (channel)
        
        Returns:
            
            array (2D ndarray):     the standardized 2D numpy array 
    '''
    
    for i in range(array.shape[1]):
        
        array[:, i] = (array[:, i] - means[i]) / stds[i]
    
    return array


def filter_DC_effect(df_signal, numtaps, cutoff, fs, drop_gyroscope):
    
    '''
    Filters out the DC coefficient in a signal
    
        Parameters:
            
            df_signal (pd.DataFrame): A signal as a dataframe
            numtaps (int):            The number of FIR taps (order of filter)
            cutoff (int):             The cutoff frequency in Hz
            fs (int):                 The frequency of the signal in Hz
        
        Returns:
            
          filt_df (pd.DataFrame):     The filtered signal as a dataframe
    '''

    # Calculate the Nyquist frequency
    nyq = 2 * cutoff / fs 
    
    # Extract the x, y, z components of the accelerometer
    sig_x = df_signal.iloc[:, 1]
    sig_y = df_signal.iloc[:, 2]
    sig_z = df_signal.iloc[:, 3]
    
    # Create the filter
    the_filter = firwin(numtaps, nyq, pass_zero = False)
    
    # Apply the filter to the individual components
    filt_sig_x = np.convolve(the_filter, sig_x, 'same')
    filt_sig_y = np.convolve(the_filter, sig_y, 'same')
    filt_sig_z = np.convolve(the_filter, sig_z, 'same')
    
    # Create the final dataframe 
    if drop_gyroscope:
        
        filt_df = pd.DataFrame(np.column_stack((df_signal['T'], filt_sig_x, filt_sig_y, filt_sig_z, df_signal['GT'])), columns = df_signal.columns)
    else: 
        
        filt_df = pd.DataFrame(np.column_stack((df_signal['T'], filt_sig_x, filt_sig_y, filt_sig_z, df_signal['GyrX'], df_signal['GyrY'], df_signal['GyrZ'], df_signal['GT'])), columns = df_signal.columns)
    
    return filt_df
    

def extract_standard_params(subjects_X):
    
    '''
    Calculates the standardization parameters for a dataset

        Parameters:

            subjects_X (dict):          A dictionary containing the whole dataset in the form of {subject: (session, bacth, window)}
            
        Returns:
            
            mean_params (1D ndarray):   An 1D array containing the desired means for each column (channel)
            std_params (1D ndarray):    An 1D array containing the desired standard deviation for each column (channel)
    '''

    huge_array = subjects_X.copy()
    
    # Create a huge numpy array with everything stacked
    for subj in huge_array:
      
        # Turn the list of lists into a 3D array (batch, window, chanels) made of the concatenated arrays on session axis 
        huge_array[subj] = np.concatenate(huge_array[subj], axis = 0)

    huge_array = np.concatenate([huge_array[subj] for subj in huge_array], axis = 0)
    huge_array = np.concatenate(huge_array, axis = 0)
    
    # Calculate the parameters from this huge array
    mean_params = np.mean(huge_array, axis = 0)
    std_params = np.std(huge_array, axis = 0)
    
    return mean_params, std_params


def augment_dataset(subjects_X, subjects_Y, drop_gyr):
    
    '''
    Artificially augments the dataset by adding rotated samples
    
        Parameters:
            
            subjects_X (dict): A dictionary containing the whole dataset in the form of {subject: (session, bacth, window)}
        
        Returns:
            
            subjects_X (dict): The augmented dataset (X) in the same form as the input
            subjects_Y (dict): The augmented dataset labels (Y) in the same form as the input
    '''

    # Augment the sessions by adding new rotated windows
    for subj in subjects_X:
    
        for sess in subjects_X[subj]:
            
            temp_sess = sess.copy()
            
            for window in temp_sess:
                
                # Rotation angles (0.1745 rads = 10 degrees)
                th_x, th_z = np.random.normal(loc = 0, scale = 0.1745, size = 2)
                
                # Rotation Matrix for X
                Rx = np.array([[1, 0, 0],
                               [0, np.cos(th_x), -np.sin(th_x)],
                               [0, np.sin(th_x), np.cos(th_x)]])
                
                # Rotation Matrix for Z
                Rz = np.array([[np.cos(th_z), -np.sin(th_z), 0],
                               [np.sin(th_z), np.cos(th_z), 0],
                               [0, 0, 1]])
                
                # Randomly choose one of four options for transformation matrix
                r = np.random.randint(low = 0, high = 4)
                
                if(r == 0):
                    
                    Q = Rx
                    
                elif(r == 1):
                    
                    Q = Rz
                    
                elif(r == 2):
                    
                    Q = Rx @ Rz
                    
                elif(r == 3):
                    
                    Q = Rz @ Rx
                
                # Create the final transformation matrix for this window
                if drop_gyr:
                    
                    final_transf = Q
                else:
                    
                    temp_1 = np.concatenate((Q, np.zeros((3, 3))), axis = 1)
                    temp_2 = np.concatenate((np.zeros((3, 3)), Q), axis = 1)
                    final_transf = np.concatenate((temp_1, temp_2), axis = 0)
                
                # Create the new augmented window and add it to the dataset
                aug_win = (final_transf @ window.T).T
                sess.append(aug_win)
    
    # Duplicate the labels of the original dataset into the augmented one
    for subj in subjects_Y:
    
        for sess in subjects_Y[subj]:
            
            sess.extend(sess)
    
    return subjects_X, subjects_Y


def calc_tp_fp_fn(test, pred, peaks):
    
    '''
    Calculates the TP, FP and FN from the predicted puffs (after their peaks have been found)
    
        Parameters:
            
            test (list of int):     The test labels (0 and 1)
            pred (list of int):     The predicted values of the class, having been set to 0 and 1
            peaks (list of int):    The calculated peaks of the predicted values before they were set to 0 and 1
        
        Returns:
            
            tp, fp, fn (int):       True-Positives, False-Positives, False-Negatives
    '''
    
    tp = 0
    fp = 0
    
    for p in peaks:
        
        # Here we allow a margin of +-5 samples to count as True Positive
        if(pred[p] == 1 and sum(test[p - 5: p + 5]) >= 1):
            
            tp = tp + 1
    
        elif(pred[p] == 1 and sum(test[p - 5 : p + 5]) == 0):
            
            fp = fp + 1
    
    # Find the peaks from the true puffs
    test_p, _ = find_peaks(test)
    
    # FN = How many True Puffs - How many predicted correctly
    fn = len(test_p) - tp
    
    return tp, fp, fn