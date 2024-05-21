# -*- coding: utf-8 -*-
"""
Created on Sun May 19 20:49:14 2024

@author: marlo
"""

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms.functional as tf
import numpy as np
import pandas as pd
import time
import os


# Verify if we are on Google Colab
try:
    import google.colab
    run_on_colabs = True
except:
    run_on_colabs = False


os.chdir(root_folder)
from model import UResNet18
from utils import *



class easy_train():
    
    def __init__(self, root_folder):
        # Defining 'self' variables
        self.root_folder = root_folder
        # If we are on Colab, mount the drive
        try:
            # Importing Drive
            from google.colab import drive
            drive.mount('/content/gdrive')
            import sys
            sys.path.append(root_folder)
        except:
            pass
    
    
    

# import configparser

# # Crie o objeto ConfigParser e carregue o arquivo .ini
# config = configparser.ConfigParser()
# config.read('config.ini')

# # Acesse os valores dos hiperparâmetros diretamente
# learning_rate = config.getfloat('Hyperparameters', 'learning_rate')
# batch_size = config.getint('Hyperparameters', 'batch_size')
# num_layers = config.getint('Hyperparameters', 'num_layers')
# # ...e assim por diante para todos os 10 hiperparâmetros

