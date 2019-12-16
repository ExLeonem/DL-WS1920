import numpy as np
import matplotlib.pyplot as plt
import activation_functions as act_f


class NeuralNet:
    """
        Fully connected neural network.
    """

    def __init__(self, input_layer, hidden_layers, output_layer):
        self.sizes = input_layer + hidden_layers + output_layer
        self.ouput_layer = output_layer # 
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]] # biases starting at first hidden layer
        self.weights = [self.__init_weights(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])] # Weights between layers
    

    def __init_weights(self, size_l, size_l1):
        """
            Initializer for the weights
        """
        return np.random.randn(size_l, size_l1) * np.sqrt(2/(size_l+size_l1))


    def __forward(self, data):
        """
            Pass data through neural network and calculate all values.
        """

        weights = self.weights
        biases = self.biases
        for i in range(len(self.weights)):
            result = np.dot(weights[i], data)
            summed_up = np.sum(result, axis = 1) + biases[i]
            data = act_f.sig(result)

        return data


    def __backward(self, data):
        """
            Calculate derivatives for each neuron.
        """
        activation = data
        activations = []

        for 
        



    def __update(self, weights, derivatives):
        pass



    def fit(self, train_data, train_labels, epochs = 150, m_batch_size = 10, l_rate = 0.03):
        """
            Fit neural network to data.

        """
    
        # Set parameters for training
        sample_count = train_data.shape[0]
        iterations = epochs * (sample_count/m_batch_size)
    
        for i in range(int(iterations)):
            batch_indices = np.random.randint(sample_count, size = (m_batch_size))
            mini_batch = train_data[batch_indices]
            fp_result = self.__forward(mini_batch)
            self.__backward(fp_result)


            break



        return 0