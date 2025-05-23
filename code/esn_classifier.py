from reservoir import Reservoir
from sklearn.metrics import accuracy_score
import numpy as np

class ESNClassifier(object):
    """Create an ESN model with a ridge regression classifier
    Parameters:
    ----------
    in_size: int
        number of input units, i.e., number of features + bias
    res_size: int
        number of reservoir units or neorons 
    trans_len: int
        trainsient length. The transient is [0, trans_len ). 
    spec_rad: float
        spectral radius: max(absolute values of the eigen values of the
        reservoir weight matrx)
    init_scale: float
        input scaling of the input and reservoir weight matrices
    lr: float
        leaking rate.
    resevoir: object, optional
        an instance of the Reservoir class
    random_gen: object, optional
        random number generator
    re: float, default = 1e-2
        regularization
    """
    def __init__(self, in_size: int, 
                 res_size: int,
                 out_size,
                 trans_len: int,
                 spec_rad: float = None,
                 init_scale: float = None,
                 lr: float = None,
                 reservoir: object = None,
                 random_gen: object = None,
                 reg: float = 1e-2
                 ):
        # initialize random number generator
        self.rng = np.random.default_rng(42) \
            if random_gen is None else random_gen
        # initialize reservoir
        if reservoir is None:
            self.reservoir = Reservoir(in_size,
                                 res_size,
                                 rand_gen=self.rng,
                                 lr = lr,
                                 spec_rad =spec_rad,
                                 init_scale = init_scale)
        else:
            self.reservoir = reservoir
        
        # initialize model parameters
        self.trans_len = trans_len
        self.reg = reg
        self.__out_size = out_size
        self.in_size = in_size
        self.res_size = res_size

        # allocate memory to store trained output weights
        self.Wout = None
        # allocate memory to store reservoir output, i.e.,
        # bias + input + reservoir state
        self.X = None
        # allocate memory to accumulate predicted outputs
        self.Y_pred = None
  
    def train(self, X_train, Y_train):
        """
        collect reservoir outputs and use them to 
        train the output weights
        
        """
        # get the size of the training set
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
        self.X = X
        X_T = self.X.T
        # compute Wout by ridge regression
        # self.readout.fit(self.X, Y_train)
        self.Wout = np.dot( np.dot(Y_train.T,X_T), np.linalg.inv( np.dot(self.X,X_T) + \
        self.reg * np.eye(1+ self.in_size + self.res_size) ) )
        print('Training completed')
        return self.Wout

    def test(self, X_test, Y_test):
        # get length of test data
        test_len = len(X_test)
        # allocate memory to collect predicted outputs
        self.Y_pred = np.zeros((self.__out_size,test_len))

        # print(f'Previous state: \n {self.reservoir.prev_res_state}')
        for t in range(test_len):
            # retrieve the first test input
            u = np.matrix(X_test[t]).T
            # update the reservoir with u. Previous state = last state after training
            state = self.reservoir.get_res_state(u)
            # predict the output for u
            # res_out =  np.vstack((1,u,state))
            # self.Y_red[:t] = readout.predict()
            y = np.dot(self.Wout, np.vstack((1,u,state)) )
            if (y > 0.5):
                self.Y_pred[:,t] = 1
            else:
                self.Y_pred[:,t] = 0

        # transpose Yhat and Y_test and change them to 1d arrays
        Yhat = self.Y_pred.T[:,0]
        Y_test = np.squeeze(np.asarray(Y_test.T))
        print('Test completed')
        return(accuracy_score(Y_test,Yhat))
   
    def set_regularization(self, Y_train, reg):
        # check if model has been trained
        if self.X is not None:
            # retrain the output weights with new regularization
            X_T = self.X.T
            # change Y_train to 2d array so that dim(Y_train) = dim(X_T)
            Y_train = np.asmatrix(Y_train)
            # reset regularization
            self.reg = reg
            # compute Wout by ridge regression
            self.Wout = np.dot( np.dot(Y_train,X_T), np.linalg.inv( np.dot(self.X,X_T) + \
            self.reg * np.eye(1+ self.in_size + self.res_size) ) )
            return (self.Wout)
        else:
            print('Model has not been trained')
  
