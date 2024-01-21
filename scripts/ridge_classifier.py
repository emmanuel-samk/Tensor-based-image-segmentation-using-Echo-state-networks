import numpy as np

class RidgeClassifier():
    """Create a ridge regression classifier
    """
    def __init__(self, regularization):
        self.reg = regularization
        self.Wout = 0.0

    def fit(self, res_out, Y_train):
        """fit a ridge regressor 
        """
        # read n_variables (bias+ in_size + res_size)
        n_variables,  = res_out.shape
        print(n_variables)
        X_T = res_out.T
        # compute Wout by ridge regression
        self.Wout = np.dot( np.dot(Y_train.T,X_T), np.linalg.inv( np.dot(self.X,X_T) + \
        self.reg * np.eye(n_variables) ) )
        # self.reg * np.eye(1+ self.in_size + self.res_size) ) )
        return self.Wout
    
    def predict(self,res_out, t):
        y = np.dot(self.Wout, res_out)
        return 1 if (y > 0.5) else 0 