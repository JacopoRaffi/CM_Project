import numpy as np
from math_utils import *

class ELM:
    '''
    Extreme Learning Machine (ELM) model for regression (linear activation function for output layer)

    Attributes
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
    init_params: tuple
        Parameters for initialization method
    init_method: str
        Initialization method for hidden layer weights
    R: np.ndarray
        R matrix for QR factorization
    h_vectors: np.ndarray
        Householder vectors for QR factorization
    X: np.ndarray
        Hidden layer output
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

        # used for incremental QR factorization
        self.R = None
        self.h_vectors = None
        self.X = None

    def fit(self, D:np.ndarray, y:np.ndarray, alfa_tikhonov:float, save_state:bool=False):
        '''
        Train the model

        Parameters
        -----------
        D: np.ndarray
            Input dataset
        y: np.ndarray
            labels
        
        Returns
        --------
        None
        '''
        
        X = self.activation(D @ self.hidden_weights.T) # hidden layer output 
        _, n = X.shape
        # vtsack under X an identity matrix  
        X = np.vstack((X, alfa_tikhonov * np.eye(n)))

        self.__solve_lstsq(X, y, save_state)

    def predict(self, D:np.ndarray):
        '''
        Predict output for given input

        Parameters
        -----------
        X: np.ndarray
            Input data
        
        Returns
        --------
        np.ndarray
            Predicted output
        '''

        if self.w is None:
            raise RuntimeError('model is not trained yet')
        
        return self.w @ self.activation(D @ self.hidden_weights.T).T

    def add_neuron(self, new_input_feature:np.ndarray, y:np.ndarray=None):
        '''
        Add a neuron to the hidden layer

        Parameters
        -----------
        new_input_feature: np.ndarray
            New input feature
        y: np.ndarray
            Labels. If None then the model will not be refitted. If not None, the model will be refitted
        
        Returns
        --------
        None
        '''

        # new random neuron
        neuron_weights = self.__init_weights(init_method=self.init_method, init_params=self.init_params, rows=1, cols=self.input_size)
        self.hidden_weights = np.vstack((self.hidden_weights, neuron_weights))

        # update X matrix
        incremental_condition = (self.X is not None) and (self.R is not None) and (self.h_vectors is not None)

        if not incremental_condition:
            raise RuntimeError('Need to save the state of the model first. Set save_state=True in fit method')

        x_new = self.activation(new_input_feature).T
        self.X = np.hstack((self.X, x_new))

        self.h_vectors, self.R = incr_QR(x_new, self.h_vectors, self.R)

        if y is not None: # refit the model
            self.__solve_lstsq(self.X, y, save_state=True)

    
    def clean_state(self):
        '''
        Clean the state of the model, i.e. set R, h_vectors and X to None so to avoid useless memory usage during inference

        Parameters
        -----------
        None
        
        Returns
        --------
        None
        '''

        self.R = None
        self.h_vectors = None
        self.X = None


    def __init_weights(self, init_method: str = 'uniform', init_params: tuple = (-1, 1), rows: int = None, cols: int = None):
        '''
        Initialize weights for hidden layer

        Parameters
        ----------
        init_method: str
            Initialization method for hidden layer weights. At the moment, only 'uniform' and 'normal' are supported
        init_params: tuple
            Parameters for initialization method
        rows: int
            Number of rows for the weight matrix (default is None)
        cols: int
            Number of columns for the weight matrix (default is None)

        Returns
        -------
        np.ndarray
            Hidden layer weights
        '''

        if init_method == 'uniform':
            return np.random.uniform(init_params[0], init_params[1], (rows, cols))
        elif init_method == 'normal':
            return np.random.normal(init_params[0], init_params[1], (rows, cols))
        else:
            raise ValueError('Invalid initialization method')
        
    def __solve_lstsq(self, X:np.ndarray, y:np.ndarray, save_state:bool=False):
        '''
        Solve the least squares problem using QR factorization

        Parameters
        -----------
        X: np.ndarray
            Matrix to be factorized
        y: np.ndarray
            Labels
        save_state: bool
            Save the state of the model (R, h_vectors, X)
        
        Returns
        --------
        np.ndarray
            Output weights
        '''

        m, n = X.shape

        if m >= n: # X tall and thin
            h_vectors, R = thin_QR(X)
            b = apply_householders_vector(h_vectors, y, reverse=False)
            
            self.w = np.squeeze(backward_substitution(R, b[:n]))
        else: # X short and wide
            h_vectors, R = thin_QR(X.T)
            z = forwad_substitution(R.T, y)
            z = np.vstack((z, np.zeros((n - m, 1))))
            
            self.w = np.squeeze(apply_householders_vector(h_vectors, z, reverse=True))
        
        if save_state:
            self.R = R
            self.h_vectors = h_vectors
            self.X = X
