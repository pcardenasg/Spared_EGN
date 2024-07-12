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
    #parser.add_argument('--sota',                       type=str,           default='pretrain',                 help='The name of the sota model to use. "None" calls main.py, "nn_baselines" calls nn_baselines.py, "pretrain" calls pretrain_backbone.py, and any other calls main_sota.py', choices=['None', 'pretrain', 'stnet', 'nn_baselines', "histogene"])
    #parser.add_argument('--img_backbone',               type=str,           default='ShuffleNetV2',             help='Backbone to use for image encoding.', choices=['resnet', 'ConvNeXt', 'MobileNetV3', 'ResNetXt', 'ShuffleNetV2', 'ViT', 'WideResNet', 'densenet', 'swin'])
    #parser.add_argument('--img_use_pretrained',         type=str2bool,      default=True,                       help='Whether or not to use imagenet1k pretrained weights in image backbone.')
    #parser.add_argument('--pretrained_ie_path',         type=str,           default='None',                     help='Path of a pretrained image encoder model to start from the contrastive model.')
    #parser.add_argument('--freeze_img_encoder',         type=str2bool,      default=False,                      help='Whether to freeze the image encoder. Only works when using pretrained model.')
    #parser.add_argument('--act',                        type=str,           default='None',                     help='Activation function to use in the architecture. Case sensitive, options available at: https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity')
    #parser.add_argument('--graph_operator',             type=str,           default='None',                     help='The convolutional graph operator to use. Case sensitive, options available at: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#convolutional-layers', choices=['GCNConv','SAGEConv','GraphConv','GATConv','GATv2Conv','TransformerConv', 'None'])
    #parser.add_argument('--pos_emb_sum',                type=str2bool,      default=False,                      help='Whether or not to sum the nodes-feature with the positional embeddings. In case False, the positional embeddings are only concatenated.')
    #parser.add_argument('--h_global',                   type=str2h_list,    default='//-1//-1//-1',             help='List of dimensions of the hidden layers of the graph convolutional network.')
    #parser.add_argument('--pooling',                    type=str,           default='None',                     help='Global graph pooling to use at the end of the graph convolutional network. Case sensitive, options available at but must be a global pooling: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#pooling-layers')
    #parser.add_argument('--dropout',                    type=float,         default=0.0,                        help='Dropout to use in the model to avoid overfitting.')
    # Train parameters #######################################################################################################################################################################
    #parser.add_argument('--noisy_training',             type=str2bool,      default=False,                      help='Whether or not to use noisy gene expression for training.')
    # *** parser.add_argument('--optim_metric',               type=str,           default='MSE',                      help='Metric that should be optimized during training.', choices=['PCC-Gene', 'MSE', 'MAE', 'Global'])
    parser.add_argument('--max_steps',                  type=int,           default=1000,                       help='Number of steps to train de model.')
    parser.add_argument('--val_check_interval',         type=int,           default=10,                         help='Number of steps to do valid checks.')
    parser.add_argument('--batch_size',                 type=int,           default=256,                        help='The batch size to train model.')
    #parser.add_argument('--shuffle',                    type=str2bool,      default=True,                       help='Whether or not to shuffle the data in dataloaders.')
    parser.add_argument('--lr',                         type=float,         default=1e-2,                       help='Learning rate to use.')
    #parser.add_argument('--optimizer',                  type=str,           default='Adam',                     help='Optimizer to use in training. Options available at: https://pytorch.org/docs/stable/optim.html It will just modify main optimizers and not sota (they have fixed optimizers).')
    #parser.add_argument('--momentum',                   type=float,         default=0.9,                        help='Momentum to use in the optimizer if it receives this parameter. If not, it is not used. It will just modify main optimizers and not sota (they have fixed optimizers).')
    #parser.add_argument('--average_test',               type=str2bool,      default=False,                      help='If True it will compute the 8 symmetries of an image during test and the prediction will be the average of the 8 outputs of the model.')
    #parser.add_argument('--cuda',                       type=str,           default='0',                        help='CUDA device to run the model.')
    parser.add_argument('--exp_name',                   type=str,           default='None',                     help='Name of the experiment to save in the results folder. "None" will assign a date coded name.')
    #parser.add_argument('--train',                      type=str2bool,      default=True,                       help='If true it will train, if false it only tests')
    ##########################################################################################################################################################################################

    return parser


# To test the code
if __name__=='__main__':
    hello = 0