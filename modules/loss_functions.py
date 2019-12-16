

def mse(result, label):
    """
        Calculates the mean-squared-error.
    """
    error = 0
    for i in range(len(result)):
        error += np.power(result[i] - label[i])

    return error / len(result) if error > 0 else error



def cross_entropy(values, weight):
    pass


def softmax(values, weight):
    pass