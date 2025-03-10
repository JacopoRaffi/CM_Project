import numpy as np

class ELM:
    '''
    Extreme Learning Machine (ELM) model for regression (linear activation function for output layer)

    Attributes:
    -----------
    n_input: int
        Number of input neurons
    n_hidden: int
        Number of hidden neurons
    n_output: int
        Number of output neurons
    activation: callable
        Activation function for hidden layer
    hidden_weights: np.ndarray
        Hidden layer weights
    w: np.ndarray
        Output weights
    '''
    def __init__(self, n_input:int, n_hidden:int, n_output:int=1, 
                 init_method:str='uniform', init_params:tuple=(-1, 1), hidden_activation:callable=np.tanh):
        
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.activation = hidden_activation

        self.hidden_weights = self.__init_weights(init_method, init_params) # hidden layer weights
        self.w = None # output weights
        
    def fit(self, X, y):
        pass

    def predict(self, X):
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
        
        return self.w @ self.activation(self.hidden_weights @ X.T)

    def __init_weights(self, init_method:str='uniform', init_params:tuple=(-1, 1)):
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
            return np.random.uniform(init_params[0], init_params[1], (self.n_hidden, self.n_input))
        elif init_method == 'normal':
            return np.random.normal(init_params[0], init_params[1], (self.n_hidden, self.n_input))
        else:
            raise ValueError('Invalid initialization method')
