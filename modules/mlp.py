import numpy as np
import math


class MLP:

    def __init__(self, arch, activation ="sigmoid", loss = "mse", epochs = 150, mbs = 10, eta = .03):
        self.arch = arch
        self.num_layers = len(arch)
        self.biases = [np.random.randn(y, 1) for y in self.arch[1:]] # biases initializieren
        self.weights = [self.__weight_init(x, y) for x, y in zip(self.arch[:-1], self.arch[1:])] # Gewichte anlegen

        # set activation function to be used
        self.activation_name = activation;
        self.activation = self.__init_activation(activation)
        self.activation_prime = self.__init_prime(activation)

        # loss config
        self.loss_name = loss
        self.loss_fun = self.__init_loss(loss)
        self.loss_count = 0

        # Training Parameter
        self.epochs = epochs
        self.mbs = mbs
        self.eta = eta


    def __backprop(self, x, y):
        """
            Return a tuple ``(nabla_b, nabla_w)`` representing the gradient for the cost function C_x.  ``nabla_b`` and
            ``nabla_w`` are layer-by-layer lists of numpy arrays, similar to ``self.biases`` and ``self.weights``.
        """
    
        # Initialisiere Updates für Schwellwerte und Gewichte
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Vorwärtslauf
        activation = x # Initialisierung a^1 = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activation(z)
            activations.append(activation)
        
        # Rückwärtslauf
        delta = self.__cost_derivative(activations[-1], y) * self.activation_prime(zs[-1]) # Fehler am Output
        self.loss_count += delta
        nabla_b[-1] = delta # Update Schwellwert in der Ausgangsschicht
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) # Update Gewichte in der Ausgangsschicht
        for l in range(2, self.num_layers): # Backpropagation
            z = zs[-l] # gewichteter Input
            sp = self.activation_prime(z) # Ableitung der Aktivierungsfunktion
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp # Fehler in Schicht l
            nabla_b[-l] = delta # Update Schwellwert 
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) # Update Gewichte

        return (nabla_b, nabla_w)


    def __feedforward(self, a):
        """
            Return the output of the network if ``a`` is input.
        """

        for b, w in zip(self.biases, self.weights):
            a = self.activation(np.dot(w, a)+b)

        return a


    def __update_mini_batch(self, xmb, ymb, eta):
        """
            Update the network's weights and biases by applying
            gradient descent using backpropagation to a single mini batch.
            The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
            is the learning rate.
        """

        # Initialisiere Updates für Schwellwerte und Gewichte
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Gehe durch alle Beispielpaare im Minibatch
        for i in range(xmb.shape[0]):
            x = np.reshape(xmb[i,:],(xmb.shape[1],1)).copy()
            if len(ymb.shape) == 2:
                y = np.reshape(ymb[i,:],(ymb.shape[1],1)).copy()
            else:
                y = ymb[i].copy()
            
            # Berechne Updates für alle Schichten über Backprop
            delta_nabla_b, delta_nabla_w = self.__backprop(x, y)
            
            # Addiere einzelne Updates auf
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        # Berechne neue Gewichte
        self.weights = [w-(eta/xmb.shape[0])*nw for w, nw in zip(self.weights, nabla_w)]

        self.biases = [b-(eta/xmb.shape[0])*nb for b, nb in zip(self.biases, nabla_b)]
        
        return (self.weights, self.biases)


    def __evaluate(self, x2, y2):
        """
            Return the number of test inputs for which the neural
            network outputs the correct result. Note that the neural
            network's output is assumed to be the index of whichever
            neuron in the final layer has the highest activation.
        """
        
        correct = 0 # Anzahl korrekt klassifizierter Testbeispiele
        mse = 0 # mean-squared-loss
        
        # Gehe den Testdatensatz durch
        for i in range(0, x2.shape[0]):
            x = np.reshape(x2[i,:],(x2.shape[1],1)).copy()
            if len(y2.shape) == 2:
                y = np.reshape(y2[i,:],(y2.shape[1],1)).copy()
            else:
                y = y2[i].copy()

            # Vorwärtslauf
            ypred = self.__feedforward(x)

            # MSE Loss summation
            mse += math.pow(y - ypred, 2)
            
            # Falls beide übereinstimmen, addiere zur Gesamtzahl
            if (int(y) == 1 and float(ypred) >= 0.5) or (int(y) == 0 and float(ypred) < 0.5) :
                correct += 1
            
        mse = mse / x2.shape[0]
        
        return (correct, mse)


    def fit(self, epochs = 150, mbs = 10, eta = .03):
        """
            Fit's the neural network to given training data.

        """
        self.epochs = epochs
        self.mbs = mbs
        self.eta = eta


    def predict(self, x):
        """
            Predict a single sample.

                x - single data sample
        """
        new_x = x.reshape(x.shape[0], 1)
        return float(self.__feedforward(new_x))


    def train(self, x0, y0, x2, y2, metrics = ["acc"]):
        """
            Train the MLP using SGD approach.

            x0 - training samples
            y0 - training labels

            x2 - test samples
            y2 - test labels

            return (accuracy, mse-loss)
        """

        print(self.__desc())

        n_test = x2.shape[0] # Anzahl Testdaten
        n = x0.shape[0]      # Anzahl Trainingsdaten

        # Get set fitting paramter
        epochs = self.epochs
        mini_batch_size = self.mbs
        eta = self.eta

        
        # gehe durch alle Epochen
        acc_val = np.zeros(epochs)
        mse_loss = np.zeros(epochs)
        for j in range(epochs):
            
            # Bringe die Trainingsdaten in eine zufällige Reihenfolge für jede Epoche
            p = np.random.permutation(n) # Zufällige Permutation aller Indizes von 0 .. n-1
            x0 = x0[p,:]
            y0 = y0[p]
            
            # Zerlege den permutierten Datensatz in Minibatches 
            for k in range(0, n, mini_batch_size):
                xmb = x0[k:k+mini_batch_size,:]
                if len(y0.shape) == 2:
                    ymb = y0[k:k+mini_batch_size,:]
                else:
                    ymb = y0[k:k+mini_batch_size]
                self.__update_mini_batch(xmb, ymb, eta)
            
            # Gib Performance aus
            (correct, mse) = self.__evaluate(x2, y2)
            acc_val[j] = correct
            mse_loss[j] = mse
            print("Epoch {0}: {1} / {2} [MSE: {3}]".format(j, acc_val[j], n_test, round(mse_loss[j], 3)))
        
        return (acc_val, mse_loss)



    # -----------------------
    # Initializers
    # -----------------------

    def __init_activation(self, function_name):
        
        if function_name == "sigmoid":
            return self.sigmoid

        elif function_name == "tanh":
            return self.tanh

        raise ArgumentError("There's currently no support for activation named " + str(function_name))


    def __init_loss(self, loss_name):

        if loss_name == "mse":
            return 


    def __init_prime(self, function_name):

        if function_name == "sigmoid":
            return self.sigmoid_prime

        elif function_name == "tanh":
            return self.tanh_prime
    
    
    def __weight_init(self, weight_prev, weight_current):
        return np.random.randn(weight_current, weight_prev) * np.sqrt(2/(weight_prev+weight_current))


    # -----------------------
    # Loss-Functions
    # -----------------------

    def mse(self, y, ypred):
        pass


    def lg_cost(self, y, ypred):
        pass


    # -----------------------
    # Activation functions
    # -----------------------

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))


    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))


    def tanh(self, z):
        return np.tanh(z);


    def tanh_prime(self, z):
        return (1.0 / np.cosh(np.cosh(z)))


    # Ableitung der MSE-Kostenfunktion
    def __cost_derivative(self, output_activations, y):
        """
            Return the vector of partial derivatives \partial C_x /
            \partial a for the output activations.
        """
        return (output_activations-y)


    # -------------------------
    # Utilities
    # -------------------------

    def __desc(self):

        # Net parameters
        epochs = self.epochs
        eta = self.eta
        mbs = self.mbs

        activation = self.activation_name
        loss = "mse"

        header = f"++++++++++++++++++\n MLP SETUP \n++++++++++++++++++\n"
        parameters = f" Epochs: {epochs}\n Learning-Rate: {eta}\n Mini-batch-size: {mbs}\n"
        neuron_setup = f"------------\n Activation: {activation}\n Loss: {loss}"

        return header + parameters + neuron_setup + "\n++++++++++++++++++"