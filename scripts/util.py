import os.path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def combine_datasets(locs):
    """ Combine image datasets at located at locs into a single dataset (mat)
    Parameters:
    ----------
    locs: list of paths
        the locations of the files to be combined
    """
    print("Processing data...")
 
    # print(files)
    # allocate memory to store combined dataset
    mat = np.empty(0)
    init_val = 0      # this initial value implies mat is empty
    for f in locs:
        # read file at location f
        df = pd.read_csv(f, sep =',',skiprows =1, header =None)
        # covert all values in df except values in column 0 to numpy matrix
        data = df.iloc[:,1:].to_numpy()
        print(f'Loaded {f}')
        # store data in mat if it is empty otherwise stack it to the
        # data already in mat
        if init_val == 0:        
            mat = data
            init_val = 1
        else:
            mat = np.concatenate((mat,data),axis=0)
    return mat

def normalize(X):
    """ Normalize numparray X using MinMax
    
    """
    # -------------------------------------------------------
    # Normalize each value of X to [0,1] using custome MinMax
    # -------------------------------------------------------
    # # retrieve the number of rows and columns of X
    # n_row, n_col= X.shape
    # # allocate memory with same shape as X to store the normalized values of X
    # normX = np.empty((n_row, n_col))
    # # Find the min and max values in each feature
    # colmax =(X.max(axis=0)).tolist()     
    # colmin =(X.min(axis=0)).tolist()  
    # for i in range(n_row):
    #     # allocate memory to store rowise feature sets
    #     row = np.empty(n_col)
    #     for j in range(n_col):
    #         val= (X[i][j]-colmin[j])/(colmax[j]-colmin[j])
    #         # print(f' X[{i},{j}]: val: {val}')
    #         row[j] = val
    #     # add row to matrix normat
    #     normX[i] = row
    # ------------------------------------------------------
    # Normalize each value X to [0,1] using sklearn MinMax
    #-------------------------------------------------------
    scaler = MinMaxScaler()
    normX = scaler.fit_transform(X)
    return (normX)

def transform_save(mat, file_name, loc):
    """
    Shuffles the dataset, normalizes the features X, and
    saves the resulting dataset as f in location loc
    
    parameters:
    -----------
    mat: numpyarray
        the data to be transformed
    file_name: string
        file name
    loc: string
        path specifying where to save the file
    """
    # shuffles and splits dataset into features X and labels Y
    np.random.shuffle(mat)
    X = mat[:,:-1]
    Y = mat[:,-1]
    # Normalizes the features set of mat
    normX = normalize(X)
    mat = np.hstack((normX, Y.reshape(np.shape(normX)[0],1)))
    # covert the normalized feature set + labels to dataframe
    df = pd.DataFrame(mat)
    # set file path
    loc = os.path.join(loc,f'{file_name}.csv')
    print("Writing data to file ...")
    df.to_csv(loc, mode ='w', header = None, index = None)
    print(f'Dataset saved to {loc}')
    return mat

def write_to_file(data, loc):
    df = pd.DataFrame(data)
    df.to_csv(loc, mode ='w', header = None, index = None)
    print(f'Dataset saved to {loc}')