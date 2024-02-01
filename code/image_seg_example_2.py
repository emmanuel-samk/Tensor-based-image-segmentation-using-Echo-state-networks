from glob import glob
from esn_classifier import ESNClassifier
from esn_svm import ESNSVM
from util import *
import pickle as pk
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn import svm

esn_config = {}
esn_config['data'] =\
    '..\\data\\processed\\data.csv'             # data file path + name
esn_config['result_loc'] = 'models\\'            # where to save outputs  
esn_config['in_size'] = 15                      # number of input units of the
                                                # the ESN reservoir
esn_config['out_size'] = 1                      # number of output units
esn_config['trans_len'] = 200                   # transient length

# =========== Reservoir RNN global parameters ==============

# range of reservoir sizes
# esn_config['res_size'] =\
#     np.array([700, 800, 900, 1000], dtype =int)    
esn_config['res_size'] = 100 
esn_config['lr'] = 0.3                          # leaking rate
# range of spectral radius
esn_config['spec_rad'] = 0.01  
# esn_config['spec_rad'] =\
#     np.array([0.01,0.03, 0.05, 0.099], dtype =float)                 
                                                
esn_config['init_scale'] = 0.5                  # weight matrix scaling value

# =========== Readout parameters =================
# create an svm classification model
esn_config['readout'] = svm.SVC(kernel='rbf', C=1.0, gamma=1000)                

print (f'ESN Classifier Configuration \n {esn_config}')


# load and convert data to numpy array
data = pd.read_csv(esn_config['data'], sep =',', header =None).to_numpy()
data = data[:1000,:]
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


# =========== Train and test an ESN WITH SVM CLASSIFIER ========
model = ESNSVM(in_size = esn_config['in_size'],
            res_size= esn_config['res_size'],
            out_size= esn_config['out_size'],
            trans_len= esn_config['trans_len'],
            spec_rad= esn_config['spec_rad'],
            init_scale=esn_config['init_scale'],
            lr = esn_config['lr'],
            readout = esn_config['readout']
            )

# Drive the ESN with X_train and train the svm readout
model.train(X_train,Y_train)

# Test the accuracy of the model
print('Test started ....')
print(f'Accuracy: {model.test( X_test, Y_test)}')
