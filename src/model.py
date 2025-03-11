import numpy as np

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
            return np.random.uniform(init_params[0], init_params[1], (self.hidden_size, self.input_size))
        elif init_method == 'normal':
            return np.random.normal(init_params[0], init_params[1], (self.hidden_size, self.input_size))
        else:
            raise ValueError('Invalid initialization method')
