import numpy as np



def sig(z):
    """
        regular simgoid distribution
    """
    return 1/1+np.exp(z)


def de_sig(z):
    """
        Derivation of a sigmoid function.
    """
    return sig(z)(1-sig(z))



def relu(z):
    return np.max(0, z)


def de_relu(z):
    return 1