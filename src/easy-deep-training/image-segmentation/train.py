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
import configparser

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
    
    def __init__(self, save_here):
        # Defining the 'self' variables for 'save_here' directory
        self.save_here = save_here
        # If we are on Colab, we need to mount the drive
        try:
            # Importing Drive
            from google.colab import drive
            # Mounting drive
            drive.mount('/content/gdrive')
            # Importing 'sys' and adding 'self.save_here' to the Colab's path
            import sys
            sys.path.append(self.save_here)
        except:
            pass
        
        # Importing configuration variables from "config.ini"
        config = configparser.ConfigParser()
        config.read(os.path.join(self.save_here, 'config.ini'))
        
        # Accessing configuration directories
        train_dir = config.get('Directories', 'train_dir').split(',')
        test_dir = 
        
        
    
    
    

# import configparser

# # Crie o objeto ConfigParser e carregue o arquivo .ini
# config = configparser.ConfigParser()
# config.read('config.ini')

# # Acesse os valores dos hiperparâmetros diretamente
# learning_rate = config.getfloat('Hyperparameters', 'learning_rate')
# batch_size = config.getint('Hyperparameters', 'batch_size')
# num_layers = config.getint('Hyperparameters', 'num_layers')
# # ...e assim por diante para todos os 10 hiperparâmetros

