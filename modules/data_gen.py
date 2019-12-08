import numpy as np
import pandas as pd




def gen_samples(dimensions, **kwargs):
"""
    Generates training/test set samples.

    ARGV:
        split - training/test split ratio
        dist - distribution
        d_range - interval from which to generate data
        split - true | false wheter to generate a test/train split


    returns tuple of ((x_train, y_train), (x_test, y_test))
"""

    
    if dist != None:
        data = np.random.uniform(d_range, dimensions)    
        

    else:
