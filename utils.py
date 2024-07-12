import numpy as np
import pandas as pd
import torchvision
import json
import squidpy as sq
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.colors as colors
import os
import torch
import wandb
from tqdm import tqdm
import argparse


# Auxiliary function to use booleans in parser
str2bool = lambda x: (str(x).lower() == 'true')
str2intlist = lambda x: [int(i) for i in x.split(',')]
str2floatlist = lambda x: [float(i) for i in x.split(',')]
str2h_list = lambda x: [str2intlist(i) for i in x.split('//')[1:]]

# Function to get global parser
def get_main_parser():
    parser = argparse.ArgumentParser(description='Code for expression prediction using contrastive learning implementation.')
    # Dataset parameters #####################################################################################################################################################################
    parser.add_argument('--dataset',                    type=str,           default='10xgenomic_human_brain',   help='Dataset to use.')
    parser.add_argument('--prediction_layer',           type=str,           default='c_t_log1p',                help='The prediction layer from the dataset to use.')
    parser.add_argument('--numk',                       type=int,           default=6,                          help='Number of neighbors for exemplar learning. Original repository numk = 6.')
    # Model parameters #######################################################################################################################################################################
    # Train parameters #######################################################################################################################################################################
    #parser.add_argument('--optim_metric',               type=str,           default='MSE',                      help='Metric that should be optimized during training.', choices=['PCC-Gene', 'MSE', 'MAE', 'Global'])
    parser.add_argument('--max_steps',                  type=int,           default=1000,                       help='Number of steps to train de model.')
    parser.add_argument('--val_check_interval',         type=int,           default=10,                         help='Number of steps to do valid checks.')
    parser.add_argument('--batch_size',                 type=int,           default=256,                        help='The batch size to train model.')
    parser.add_argument('--lr',                         type=float,         default=1e-2,                       help='Learning rate to use.')
    parser.add_argument('--exp_name',                   type=str,           default='None',                     help='Name of the experiment to save in the results folder. "None" will assign a date coded name.')
    ##########################################################################################################################################################################################

    return parser


# To test the code
if __name__=='__main__':
    hello = 0