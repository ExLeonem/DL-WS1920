import numpy as np
import matplotlib.pyplot as plt


class FirstNN:
    """
        A simple neuroal network consisting of 
    """

    def __init__(self, input_dim, hidden_layers, output_dim):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim


    def train(self, data, labels, l_rate = 0.3, epochs = 150, m_batch = 10):
        
        data_set_x = data.shape[0]
        for i in range(epochs):
            rand_indx = np.random.randint(data_set_x, size = (m_batch)) # select random indices for training

            break


    def __forward(self):

        # Calcualte neuron values in forward pass for each layer
        

        pass


    def __backward(self):
        pass


    def __backpropagation(self):
        pass
