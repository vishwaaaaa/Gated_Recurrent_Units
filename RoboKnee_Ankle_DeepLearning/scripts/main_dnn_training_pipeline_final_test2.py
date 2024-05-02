"""

This is the main pipeline for building, training, and testing the DNNs. 

@author: Chris Prasanna

"""

from src import set_parameters
from src import data_processing
from src import optuna_functions
from src import testing_and_evaluation
from src import save_results

import pickle
import torch

# %% Load experimental data

with open('../Data_train/datatrain2new_structure.pkl', 'rb') as f:
    data_structure = pickle.load(f)
del f

with open('../Data_train/test_newstructure.pkl', 'rb') as f:
    test_structure = pickle.load(f)
del f

# %% Define Constants and Hyperparameters

# Set Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


# This function gets all constants used in the script (metadata, filepaths,
# number of trials, timesteps per trial, etc.)
constants = set_parameters.get_constants(device=device, input_size=15, output_size=2, optuna_trials=1, num_timestepsPerTrial=4000, fs=160)

# This function gets all hyperparameter ranges used in the Optuna optimization
hyperparameter_ranges = set_parameters.get_hyperparameter_ranges()

# %% Cut data into train vs test sets

train_set, validation_set, test_set1 = data_processing.train_val_test_split_data(data_structure, split_percentages=[0.80,0.20,0.00])

train_set_names = test_structure['file names']

test_set = {}
for name in train_set_names:
    test_set[name] = test_structure['data'][name]

# %% Choose and Train Model(s)

model_types = ['GRU']#'FFN', 'DA-GRU', 
test_results_dict = {}

for model_type in model_types:
                                                                           
    ## Optimize Hyperparameters
    optimized_model, best_trial = optuna_functions.optimize_hyperparameters(
        model_type=model_type, train_set=train_set, val_set=validation_set, constants=constants, hyperparameter_ranges=hyperparameter_ranges)

    ## Retrieve sequence length from best neural network
    sequence_length = best_trial.params['sequence length']
    
    ## Compute & visualize test results
    test_results = testing_and_evaluation.main_test(model=optimized_model, model_type=model_type, data_set=test_set, constants=constants, sequence_length=sequence_length)
    
    ## Save Data / Results
    save_results.main_save(model_type=model_type, optimized_model=optimized_model, test_results=test_results)
    
    ## Store Results to Dictionary
    test_results_dict[model_type] = test_results
    with open('test1_GRUAB05_2activit_results_dict.pickle', 'wb') as f:
        pickle.dump(test_results_dict, f)

print("\n\n*** Finished ***")