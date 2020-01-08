import numpy as np

class DemoNN:

    def __init__(self, eta):
        self.eta = eta


    def backprop(self, x, y):
        """
            Return a tuple ``(nabla_b, nabla_w)`` representing the
            gradient for the cost function C_x.  ``nabla_b`` and
            ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
            to ``self.biases`` and ``self.weights``.
        """
        
        # Initialisiere Updates für Schwellwerte und Gewichte
        nabla_b = [np.zeros(b.shape) for b in biases]
        nabla_w = [np.zeros(w.shape) for w in weights]
        
        # Vorwärtslauf
        activation = x # Initialisierung a^1 = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(biases, weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        
        # Rückwärtslauf
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1]) # Fehler am Output
        nabla_b[-1] = delta # Update Schwellwert in der Ausgangsschicht
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) # Update Gewichte in der Ausgangsschicht
        for l in range(2, num_layers): # Backpropagation
            z = zs[-l] # gewichteter Input
            sp = self.sigmoid_prime(z) # Ableitung der Aktivierungsfunktion
            delta = np.dot(weights[-l+1].transpose(), delta) * sp # Fehler in Schicht l
            nabla_b[-l] = delta # Update Schwellwert 
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) # Update Gewichte

        return (nabla_b, nabla_w)


    def update_mini_batch(self, xmb, ymb, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        global weights
        global biases
        eta = self.eta

        # Initialisiere Updates für Schwellwerte und Gewichte
        nabla_b = [np.zeros(b.shape) for b in biases]
        nabla_w = [np.zeros(w.shape) for w in weights]
        
        # Gehe durch alle Beispielpaare im Minibatch
        for i in range(xmb.shape[0]):
            x = np.reshape(xmb[i,:],(xmb.shape[1],1)).copy()
            if len(ymb.shape) == 2:
                y = np.reshape(ymb[i,:],(ymb.shape[1],1)).copy()
            else:
                y = ymb[i].copy()
            
            # Berechne Updates für alle Schichten über Backprop
            delta_nabla_b, delta_nabla_w = backprop(x, y)
            
            # Addiere einzelne Updates auf
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        # Berechne neue Gewichte
        weights = [w-(eta/xmb.shape[0])*nw for w, nw in zip(weights, nabla_w)]
        biases = [b-(eta/xmb.shape[0])*nb for b, nb in zip(biases, nabla_b)]
        
        return (weights, biases)


    def evaluate(self, x2, y2):
        """
            Return the number of test inputs for which the neural
            network outputs the correct result. Note that the neural
            network's output is assumed to be the index of whichever
            neuron in the final layer has the highest activation.
        """
        
        correct = 0 # Anzahl korrekt klassifizierter Testbeispiele
        
        # Gehe den Testdatensatz durch
        for i in range(0, x2.shape[0]):
            x = np.reshape(x2[i,:],(x2.shape[1],1)).copy()
            if len(y2.shape) == 2:
                y = np.reshape(y2[i,:],(y2.shape[1],1)).copy()
            else:
                y = y2[i].copy()
            
            # Vorwärtslauf
            ypred = self.feedforward(x)
            
            # Label ist in one-hot-Codierung: korrekte Klasse ist 1, alle anderen 0
            c = np.argmax(y)
            # Index des maximal aktivierten Outputs ist die Entscheidung des Netzwerk
            cpred = np.argmax(ypred)
            
            # Falls beide übereinstimmen, addiere zur Gesamtzahl
            if (int(y) == 1 and float(ypred) >= 0.5) or (int(y) == 0 and float(ypred) < 0.5) :
                correct += 1
            
        return correct
    

    def SGD(self, x0, y0, epochs, mini_batch_size, eta, x2, y2):

        n_test = x2.shape[0] # Anzahl Testdaten
        n = x0.shape[0]      # Anzahl Trainingsdaten
        
        # gehe durch alle Epochen
        acc_val = np.zeros(epochs)
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
                self.update_mini_batch(xmb, ymb, eta)
            
            # Gib Performance aus
            acc_val[j] = evaluate(x2, y2)
            print("Epoch {0}: {1} / {2}".format(j, acc_val[j], n_test))
        
        return acc_val

    
    def sigmoid(self, z):
        """The sigmoid function."""
        return 1.0/(1.0+np.exp(-z))

    # Ableitung des Sigmoids
    def sigmoid_prime(self, z):
        """Derivative of the sigmoid function."""
        return sigmoid(z)*(1-sigmoid(z))

    # Ableitung der MSE-Kostenfunktion
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(biases, weights):
            a = sigmoid(np.dot(w, a)+b)
        return a