
import numpy as np

class Reservoir:
    """Create a reservoir RNN
    parameters
    ----------
        in_size: int
            the number of input units (determined by the number of features of the input)
        res_size: int
            the number of hidden neuorons
        lr: float, optional, default = 0.3
            leaking rate. A value in the range (0, 1], that determins
            the speed of the reservoir update dynamics. The default is None
        rand_gen: 
            random number generator
        spec_rad: float, optional, default = 0.1
            scaling factor of the spectral radius of the internal weight matrix
        init_scale: float, optional, default = 0.5
            scaling factor of the input weight matrix
        
 
    """
    def __init__(self, in_size: int, 
                 res_size:int, 
                 rand_gen: object,
                 lr =0.3, 
                 spec_rad = 0.1, 
                 init_scale =0.5):
        
        self.in_size = in_size
        self.res_size = res_size
        self.spec_rad = spec_rad
        self.lr = lr

        # generate uniformly distributed input weights 
        self.W_in = rand_gen.uniform(-init_scale,init_scale, size =(res_size,
                                                      (in_size +1)))
        # initialize the reservoir weight matrix 
        self.W = self.set_reservoir_weights(res_size, spec_rad, init_scale,rand_gen)

         # iniliaze the previous reservoir neuron activations to zeros
        self.prev_res_state = np.zeros((self.res_size, 1))

    def set_reservoir_weights(self, res_size, spec_rad, init_scale, rng):
        # generate random  weights from a uniform distribution 
        W = rng.uniform(-init_scale,init_scale, size =(res_size,
                                                      res_size))
        print ('Computing spectral radius...')
        rhoW = np.max(np.abs(np.linalg.eig(W)[0]))
        # scale the spectral radius of the reservoir weight matrix
        W *= spec_rad / rhoW
        # print(f'W: {W}')
        return W

    
    def get_res_state(self, u):
        """ Update the reservoir state
        """
        # update the reservoir neuron activations
        res_state = (1-self.lr) * self.prev_res_state + \
            self.lr * np.tanh(np.dot(self.W_in, np.vstack((1,u))) + \
                                       np.dot( self.W, self.prev_res_state ))
        # update the previous state to the current state
        self.prev_res_state = res_state
        return res_state