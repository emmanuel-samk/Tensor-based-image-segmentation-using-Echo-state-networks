from reservoir import Reservoir
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np

class ESNSVM(object):
    """Create an ESN model with svm readout
    Parameters:
    ----------
    

    """
    def __init__(self, in_size: int, 
                 res_size: int,
                 out_size,
                 trans_len: int,
                 spec_rad: float = None,
                 init_scale: float = None,
                 lr: float = None,
                 reservoir: object = None,
                 readout: object = None,
                 random_gen: object = None,
                 ):
        # set random number generator
        self.rng = np.random.default_rng(42) \
            if random_gen is None else random_gen
        
        if reservoir is None:
            self.reservoir = Reservoir(in_size,
                                 res_size,
                                 rand_gen=self.rng,
                                 lr = lr,
                                 spec_rad =spec_rad,
                                 init_scale = init_scale)
        else:
            self.reservoir = reservoir
        
        # memory for model parameters
        self.trans_len = trans_len
        self.in_size = in_size
        self.res_size = res_size

        # # memory to store trained output weights
        # self.Wout = None
        # # memory to store collected reservoir states
        # self.X = None
        # memory to accumulate reservoir output of X_test
        self.res_out = None
        # set readout
        if readout is None:
            self.readout = svm.SVC(kernel='rbf', C=1.0, gamma=1000) 
        else:
            self.readout = readout

    def train(self, X_train, Y_train):
        """
        collect reservoir states and use it to 
        train the output weights
        
        """
        # get the length of the input data
        train_len  =  len(X_train)
        #Allocate memory to collect reservoir states after initial transient phase
        X = np.zeros((1 + self.in_size + self.res_size, train_len - self.trans_len))
        print('Fetching reservoir states ...')
        for t in range(train_len):
            # access input data at time t
            u = np.matrix(X_train[t]).T
            # # feed input at time t into the reservoir
            state = self.reservoir.get_res_state(u)
            # collect the input + bias + reservoir neuron activations 
            # when the transient time passes(before trans_len)
            if t >= self.trans_len:
                X[:,t - self.trans_len] = np.vstack((1,u,state))[:,0].T 
        # store esn feature representation
        self.X = X
        X_T = self.X.T
        # print(f'X_T: {X_T.shape}')
        # train the svm model
        self.readout.fit(X_T, Y_train)
        # print(f'Y_train: {Y_train.shape}')
        print('Training completed')

    def test(self, X_test, Y_test):
        # get length of test data
        test_len = len(X_test)
        # allocate memory to collect predicted outputs
        self.res_out = np.zeros((1 + self.in_size + self.res_size, test_len))
        # print(f'Previous state: \n {self.reservoir.prev_res_state}')
        for t in range(test_len):
            # retrieve the first test input
            u = np.matrix(X_test[t]).T
            # update the reservoir with u. Previous state = last state after training
            state = self.reservoir.get_res_state(u)
            # reservoir output for u
            self.res_out[:,t] = np.vstack((1,u,state))[:,0].T

        res_out_T = self.res_out.T
        # print(f'res_out_T {res_out_T.shape}')
        # transpose Yhat and Y_test and change them to 1d arrays
        Yhat = self.readout.predict(res_out_T)
        Y_test = np.squeeze(np.asarray(Y_test.T))
        print(Y_test.shape)
        # print('Test completed')
        return(accuracy_score(Y_test,Yhat))
  
