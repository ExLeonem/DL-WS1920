import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras




def export_model(model, file_path):
    """
        Saves a trained deep learning model into a .json file.
    """

    (ext, file_name, dir_path) = __split_path(file_path)

    # Raise error if arugment is not an
    if !__is_json(ext):
        raise ArgumentError("Model can sonly be saved in .json file.")


    weight_file = os.path.join(dir_path, file_name, "h5") # save weights into hdf5-file
    model_json = model.to_json()
    with open(file_path, 'w') as model_file:
        model_file.write(model_json)
        
    
        




def load_model(file_path):
    """
        Retrieves an saved model from a .json file.
    """

    if !__is_json(file_path):
        raise ArgumentError("Keras Models can only be loaded from a .json file")

    

# -----------------------------
# File-System utilities
# -----------------------------

def __split_path(file_path):
    """
        Splits the given file path for further processing.
    """

    _r, extension = file_path.splitext(file_path)

    splitted_rest = os.path.split(_r)
    rest_path = ""
    file_name = _r
    if len(splitted_rest) > 1:
        rest_path = os.path.join(splitted_rest[:-1])
        file_name = splitted_rest[-1]


    return (extension, file_name, rest_path)



def __path_exists(directory_path):
    """
        Check if given directory does already exists.
    """

    if os.path.isdir(directory_path):
        return True

    else:
        return False



def __is_json(extension):

    if extension == '.json':
        return True

    else:
        return False
