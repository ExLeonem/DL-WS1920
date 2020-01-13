import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import model_from_json


def compare_models(first, second):
    print("compared models")


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
        print("Model written into json.")

    model.save_weights(os.path.join(dir_path, file_name + ".h5")) # save file parameter in h5 file
    print("Weights written into file.")



def import_model(file_path):
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
    model_json_file.close()
    model = model_from_json(model_data)
    print("Model loaded")

    # Load Parameter
    model.load_weights(os.path.join(dir_path, file_name + ".h5"))
    print("Parameters loaded")

    return model
    


def compare(f_model, s_model, x_test, y_test):
    """
        Compares two models.

        @param f_model - first model
        @param s_model - second model
        @param x_test - x test data to evaluated the models on
        @param y_test - y labels to validate the predictions

        @return -1: f_model better | 0: models are equally good | 1: s_model is better
    """


    eval_first = f_model.evaluate(x_test, y_test)
    eval_sec = s_model.evaluate(x_test, y_test)

    if eval_first[-1] > eval_sec[-1]:
        return -1
    elif eval_first[-1] < eval_sec[-1]:
        return 1
    
    return 0




def model_exists(file_path):

    ext, file_name, dir_path = __split_path(file_path)

    if not __is_json(ext): 
        raise ArgumentError("Models are onyl saved in json files.")

    if os.path.exists(file_path) and os.path.isfile(file_path):
        return True

    return False



# -----------------------------
# File-System utilities
# -----------------------------

def __split_path(file_path):
    """
        Splits the given file path for further processing.
    """

    _r, extension = os.path.splitext(file_path)

    splitted_rest = os.path.split(_r)
    rest_path = ""
    file_name = _r
    if len(splitted_rest) > 1:
        rest_path = splitted_rest[0]
        file_name = splitted_rest[1]


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
