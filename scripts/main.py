from glob import glob
from esn_classifier import ESNClassifier
from util import *
import pickle as pk
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score

# ================= DATA PREPROCESSING ================
# # Retrieve all files with extension .csv in subfolder raw
# files = glob('..\\data\\raw\\*.csv')
# # Combine the image datasets into one dataset
# ds  = combine_datasets(files)
# # Shuffle the combined dataset, normalize the feature set and save the result
# transform_save(ds, 'data','..\\data\\processed')


# ================ ESN Configuration =====================
esn_config = {}
esn_config['data'] =\
    '..\\data\\processed\\data.csv'             # data file path + name
esn_config['result_loc'] = 'models\\'            # where to save outputs  
esn_config['in_size'] = 15                      # number of input units of the
                                                # the ESN reservoir
esn_config['out_size'] = 1                      # number of output units
esn_config['trans_len'] = 200                   # transient length

# Reservoir RNN global parameters 
esn_config['res_size'] = 700                    # reservoir size  
esn_config['lr'] = 0.3                          # leaking rate
esn_config['spec_rad'] = 0.9                    # spectral radius                                                     
esn_config['init_scale'] = 0.5                  # weight matrix scaling value


# ============= Readout parameters =================
esn_config['reg_coeff'] = 1e-2                  # regularization coefficient

print (f'ESN Classifier Configuration \n {esn_config}')


# ================= LOAD & SPLIT DATA ====================
# load and convert data to numpy array
data = pd.read_csv(esn_config['data'], sep =',', header =None).to_numpy()
print(f"{esn_config['data']} loaded")

# get the number of instances and variables of the dataset
n_rows, n_cols = data.shape 
# split data into train and test sets
train_len = int(n_rows * 0.8)                   # length of train set
test_len = n_rows - train_len                   # length of test set

# training input signal and respective target output signal
X_train = data[0:train_len, : -1]                     
Y_train = data[esn_config['trans_len'] :train_len, -1]  

# test input signal and respective target output signal
X_test = data[train_len: train_len + test_len, : -1]                       
Y_test = data[train_len : train_len + test_len, -1]     

print(f'Shape of training set: {X_train.shape}')
print(f'Shape of test set: {X_test.shape} ')

# ================== TRAIN & TEST AN ESN MODEL =================
# Create an ESN Classifier
model = ESNClassifier(in_size = esn_config['in_size'],
            res_size= esn_config['res_size'],
            out_size= esn_config['out_size'],
            trans_len= esn_config['trans_len'],
            spec_rad= esn_config['spec_rad'],
            init_scale=esn_config['init_scale'],
            lr = esn_config['lr'],
            reg = esn_config['reg_coeff'])

# Fit an ESN classifier to the dataset
model.train(X_train,Y_train)

# Test the accuracy of the model
print('Test started ....')
print(f'Accuracy: {model.test( X_test, Y_test)}')

# # ============ ADJUST REGULARIZATION ========================

# # Save the model to avoid reruning it through the data
# model_output = 'models\\esn_reg_model.pk'
# with open(model_output, 'wb') as file:  
#     pk.dump(model, file)

# # select a new regularization 
# model.set_regularization(Y_train, 1e-3)
# # Test the accuracy of the model
# print('Test started ....')
# print(f'Accuracy: {model.test( X_test, Y_test)}')
