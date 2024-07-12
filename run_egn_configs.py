import json
import subprocess
import argparse
import os

# Get parsed the path of the config file
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',             type=str,      default='villacampa_kidney_organoid',      help='Path to the .json file with the configs of the dataset.')
parser.add_argument('--lr',                  type=str,      default=1e-2,                              help='Learning Rate.')
args = parser.parse_args()

# Create comand to build exemplars for the dataset in dataset_config
command_list = ['python', 'exemplars.py']

# Call subprocess for exemplars
command_list.append('--dataset')
command_list.append(args.dataset)

print("Running exemplars.py ...")
subprocess.call(command_list)

# Create comand to train EGN
command_list = ['python', 'main.py']

command_list.append('--dataset')
command_list.append(args.dataset)
command_list.append('--lr')
command_list.append(args.lr)

print("Running main.py ...")
# Call subprocess for model training
subprocess.call(command_list)
