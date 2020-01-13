import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import model_from_json


def export_model(model, file_path):
    """
        Saves a trained deep learning model into a .json file.


        @param file_path - path to the .json file including the file-name.json
    """

    ext, file_name, dir_path = __split_path(file_path)

    # Save model only into .json files
    if not __is_json(ext):
        raise ArgumentError("Model can sonly be saved in .json file.")

    # Direcotry non-existent
    if not os.path.isdir(dir_path):
        raise ArgumentError("Directory: " + dir_path + " does not exist")

    # Parse model to json and write to file
    model_json = model.to_json()
    with open(file_path, 'w') as model_file:
        model_file.write(model_json)
        
    model.save_weights(os.path.join(dir_path, file_name, "h5")) # save file parameter in h5 file



def load_model(file_path):
    """
        Retrieves an saved model from a .json file.

        @param file_path - path to the model file
        @return model
    """

    ext, file_name, dir_path = __split_path(file_path)

    # Load model onyl from .json files
    if not __is_json(ext):
        raise ArgumentError("Keras Models can only be loaded from a .json file")

    # Non-exitent directory
    if not os.path.isdir(dir_path):
        raise ArgumentError("Directory : " + dir_path + " does not exist.")
    
    # Load model
    model_json_file = open(file_path)
    model_data = model_json_file.read()
    model = model_from_json(model_data)

    # Load Parameter
    model.load_weights(os.path.join(dir_path, file_name, "h5"))

    return model
    


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
