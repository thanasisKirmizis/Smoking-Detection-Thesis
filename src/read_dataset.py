# Imports
import pickle
from auxiliary_functions import filter_DC_effect, get_windows, get_window_labels

def read_dataset(name, cutoff, windows_size, windows_step, epsilon, drop_gyroscope, filter_gravity):

    # Define directories names
    folder = '../datasets'
    
    if drop_gyroscope:
        
        windows_folder = '../windows/no gyro'
        subject_folder = '../subjects/no gyro'
    else:
        
        windows_folder = '../windows/with gyro'
        subject_folder = '../subjects/with gyro'
    
    # Read selected dataset
    with open(folder + '/' + name, 'rb') as f:
        dataset = pickle.load(f)
     
    # For each subject in the dataset
    for subject in dataset:
        
        subj_dict = {}
        subj_dict['subj'] = subject
        with open(subject_folder + '/subj_'+ subject, 'wb') as f:
            pickle.dump(subj_dict, f)
        
        # For each session of the subject
        for i, session in enumerate(dataset[subject]):
            
            # If needed, drop the gyroscope measurements and work only with accelerometer
            if drop_gyroscope:
                session = session.drop(columns = ["GyrX", "GyrY", "GyrZ"])
            
            # If needed, remove gravity effect from accelerometer through a high pass filter with selected cutoff
            if filter_gravity:
                session = filter_DC_effect(session, 513, cutoff, 50, drop_gyroscope)
            
            # Extract windows from the measurements using the sliding-window technique 
            windows = get_windows(session, windows_size, windows_step)
            
            # Extract the labels of the windows using the "end-of-window" techinque with the selected epsilon
            ground_truth = list(session['GT'])
            windows_class = get_window_labels(ground_truth, windows_size, windows_step, epsilon)
            windows_class = windows_class[:len(windows)]
            
            # Save the windows and windows labels to pickle files for later use
            with open(windows_folder + '/windows_subj_' + subject + '_sess_' + str(i + 1), 'wb') as f:
                pickle.dump(windows, f)
            with open(windows_folder + '/windows_labels_subj_' + subject + '_sess_'+ str(i + 1), 'wb') as f:
                pickle.dump(windows_class, f)