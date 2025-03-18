import numpy as np
from math_utils import *

class ELM:
    '''
    Extreme Learning Machine (ELM) model for regression (linear activation function for output layer)

    Attributes:
    -----------
    input_size: int
        Number of input neurons
    hidden_size: int
        Number of hidden neurons
    output_size: int
        Number of output neurons
    activation: callable
        Activation function for hidden layer
    hidden_weights: np.ndarray
        Hidden layer weights
    w: np.ndarray
        Output weights
    '''
    def __init__(self, input_size:int, hidden_size:int, output_size:int=1, 
                 init_method:str='uniform', init_params:tuple=(-1, 1), hidden_activation:callable=np.tanh):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = hidden_activation
        self.init_params = init_params
        self.init_method = init_method

        self.hidden_weights = self.__init_weights(init_method, init_params, hidden_size, input_size) # hidden layer weights
        self.w = None # output weights

    def fit(self, D, y):
        '''
        Train the model

        Parameters:
        -----------
        D: np.ndarray
            Input dataset
        y: np.ndarray
            labels
        
        Returns:
        --------
        None
        '''
        
        X = self.activation(D @ self.hidden_weights.T) # hidden layer output 
        m, n = X.shape

        if m >= n: # X tall and thin
            h_vectors, R = thin_QR(X)
            b = apply_householders_vector(h_vectors, y, reverse=False)

            self.w = np.squeeze(backward_substitution(R, b[:n]))
        else: # X short and wide
            h_vectors, R = thin_QR(X.T)
            z = forwad_substitution(R.T, y)
            #z = np.vstack((z, np.zeros((n - m, 1))))

            self.w = apply_householders_vector(h_vectors, z[:m], reverse=True)

    def predict(self, D):
        '''
        Predict output for given input

        Parameters:
        -----------
        X: np.ndarray
            Input data
        
        Returns:
        --------
        np.ndarray
            Predicted output
        '''

        if self.w is None:
            raise RuntimeError('model is not trained yet')
        
        return self.w @ self.activation(D @ self.hidden_weights.T).T

    def add_neuron(self, new_input_feature:np.ndarray):
        '''
        Add a neuron to the hidden layer

        Parameters:
        -----------
        new_input_feature: np.ndarray
            New input feature
        
        Returns:
        --------
        None
        '''

        # new random neuron
        neuron_weights = self.__init_weights(init_method=self.init_method, init_params=self.init_params, rows=1, cols=self.input_size)


    def __init_weights(self, init_method:str='uniform', init_params:tuple=(-1, 1), rows:int=None, cols:int=None):
        '''
        Initialize weights for hidden layer

        Parameters:
        -----------
        init_method: str
            Initialization method for hidden layer weights. At the moment, only 'uniform' and 'normal' are supported
        init_params: tuple
            Parameters for initialization method
        
        Returns:
        --------
        np.ndarray
            Hidden layer weights
        '''

        if init_method == 'uniform':
            return np.random.uniform(init_params[0], init_params[1], (rows, cols))
        elif init_method == 'normal':
            return np.random.normal(init_params[0], init_params[1], (rows, cols))
        else:
            raise ValueError('Invalid initialization method')
