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

with open('../Data_train/datatrain2_structure.pkl', 'rb') as f:
    data_structure = pickle.load(f)
del f

with open('../Data_train/test_structure.pkl', 'rb') as f:
    test_structure = pickle.load(f)
del f

# %% Define Constants and Hyperparameters

# Set Device
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


# This function gets all constants used in the script (metadata, filepaths,
# number of trials, timesteps per trial, etc.)
constants = set_parameters.get_constants(device=device, input_size=15, output_size=2, optuna_trials=2, num_timestepsPerTrial=4000, fs=120)

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
    # optimized_model, best_trial = optuna_functions.optimize_hyperparameters(
    #     model_type=model_type, train_set=train_set, val_set=validation_set, constants=constants, hyperparameter_ranges=hyperparameter_ranges)


    class GRU(torch.nn.Module):

        def __init__(self):#, hyperparameters, constants
            super(GRU, self).__init__()

            # Unpack relevant hyperparameters and dimensions
            self.input_size = 15#constants['input size']
            self.output_size = 2#constants['output size']
            self.hidden_size = 512#hyperparameters['hidden size']
            self.num_layers = 10#hyperparameters['number of layers']

            # Dropout layer not included on last GRU layer
            if self.num_layers > 1:
                self.gru = torch.nn.GRU(self.input_size, self.hidden_size,
                                self.num_layers, batch_first=True,
                                dropout=0.03)#hyperparameters['dropout']
            else:
                self.gru = torch.nn.GRU(self.input_size, self.hidden_size,
                                self.num_layers, batch_first=True)
            # Define final output layer
            self.fc = torch.nn.Linear(self.hidden_size, self.output_size)

            # Define activation function
            self.relu = torch.nn.ReLU()

            # Define computing device
            self.device = 'cpu'#constants['device']

        def forward(self, x, h0):
            # forward pass through gru
            self.gru.flatten_parameters()
            out, h0 = self.gru(x, h0)

            # forward pass through output layer and activation function
            out = self.fc(self.relu(out))
            return out, h0

        def init_hidden(self, batch_size):
            # retrieve first parameter in the gru model
            weight = next(self.parameters()).data
            # creates a hidden state tensor that has the same data type and device
            # as model parameter
            hidden = weight.new(self.num_layers, batch_size,
                                self.hidden_size).zero_().to(self.device)
            return hidden




    optimized_model = GRU()
    optimized_model.load_state_dict(torch.load('../results/model1/GRU2output_trained_model.pt'))
    optimized_model.eval()
    ## Retrieve sequence length from best neural network
    sequence_length = 10#best_trial.params['sequence length']
    
    ## Compute & visualize test results
    test_results = testing_and_evaluation.main_test(model=optimized_model, model_type=model_type, data_set=test_set, constants=constants, sequence_length=sequence_length)
    
    ## Save Data / Results
    save_results.main_save(model_type=model_type, optimized_model=optimized_model, test_results=test_results)
    
    ## Store Results to Dictionary
    test_results_dict[model_type] = test_results
    with open('test1_GRUAB05_allact_results_dict.pickle', 'wb') as f:
        pickle.dump(test_results_dict, f)

print("\n\n*** Finished ***")