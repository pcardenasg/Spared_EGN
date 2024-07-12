import os
from torch.utils import data
import pytorch_lightning as pl
from functools import partial
import torch
import collections
from train import TrainerModel
import torchvision
import wandb
from datetime import datetime

from dataset import OrganizeDataset
from utils import *
from pytorch_lightning.callbacks import ModelCheckpoint

# Using spare library
from spared.datasets import get_dataset

os.environ['WANDB_CONFIG_DIR'] = os.path.join(os.getcwd(), 'wandb') 
os.environ['WANDB_DIR'] = os.path.join(os.getcwd(), 'wandb')
os.environ['WANDB_CACHE_DIR'] = os.path.join(os.getcwd(), 'wandb')

parser = get_main_parser()
args = parser.parse_args()
use_cuda = torch.cuda.is_available()

mean = [0.5476, 0.5218, 0.6881]
std  = [0.2461, 0.2101, 0.1649]

if args.exp_name == 'None':
    args.exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
results_path = f"Results_EGN/{args.dataset}/{args.exp_name}"

max_min_dict = {'PCC-Gene': 'max', 'PCC-Patch': 'max', 'MSE': 'min', 'MAE': 'min', 'R2-Gene': 'max', 'R2-Patch': 'max', 'Global': 'max'}

def main(dataset, data_batch_size, learning_r, args):
    cwd = os.getcwd()
    
    def write(director, name, *string):
        string = [str(i) for i in string]
        string = " ".join(string)
        with open(os.path.join(director,name),"a") as f:
            f.write(string + "\n")   
    
    store_dir = os.path.join(results_path)
    print = partial(write, cwd, results_path + "/" + "log_train") 
        
    os.makedirs(store_dir, exist_ok= True)
    
    transform = torchvision.transforms.Normalize(mean=mean, std=std)
    
    print(f">>>> EGN Model run with {args.dataset} dataset <<<<")
    
    # Get dataset from the values defined in args
    dataset = get_dataset(args.dataset)

    # Declare train and test datasets
    train_split = dataset.adata[dataset.adata.obs['split']=='train']
    val_split = dataset.adata[dataset.adata.obs['split']=='val']
    test_split = dataset.adata[dataset.adata.obs['split']=='test']

    test_available = False if test_split.shape[0] == 0 else True

    train_dataset = OrganizeDataset(train_split, args, transform, os.path.join('EGN_exemplars', args.dataset), 'train', pred_layer=args.prediction_layer)
    valid_dataset = OrganizeDataset(val_split, args, transform, os.path.join('EGN_exemplars', args.dataset), 'val', pred_layer=args.prediction_layer, op_data=train_split)

    # Declare data loaders
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=data_batch_size,
        num_workers = 6,
        pin_memory=True,
        persistent_workers=True,
        drop_last = True
    )
    
    val_loader = data.DataLoader(
        valid_dataset,
        batch_size=data_batch_size,
        num_workers = 6,
        pin_memory=True,
        persistent_workers=True,
        drop_last = True
    )
    
    if test_available:
        test_dataset = OrganizeDataset(test_split, args, transform, os.path.join('EGN_exemplars', args.dataset), 'test', pred_layer=args.prediction_layer, op_data=train_split)
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=data_batch_size,
            num_workers = 6,
            pin_memory=True,
            persistent_workers=True,
            drop_last = True
        )

    filter_name = dataset.adata.var_names.tolist()

    # TODO: set the parameters as parser flags for customizing options 
    # The following parameters are the ones suggested in the original EGN repository train command
    model = EGN(
        image_size = 256,
        dim = 1024,
        depth = 8,
        heads = 16,
        mlp_dim = 4096,
        bhead = 8,
        bdim = 64,
        bfre = 2,
        mdim = 2048,
        player = 2,
        linear_projection = True,
        num_classes = train_split.shape[1]
        )
    
    print(f">>  Dataset: {dataset}  <<")
    print(f">>  Batch Size: {data_batch_size}  <<")
    print(f">>  LR: {learning_r}  <<")
    print(f">>  Predict Layer: {args.prediction_layer}  <<")

    CONFIG = collections.namedtuple('CONFIG', ['lr', 'logfun', 'verbose_step', 'weight_decay', 'store_dir', 'filter_name', 'max_steps'])
    config = CONFIG(learning_r, print, 10, 0.0001, store_dir, filter_name, args.max_steps)
        
    checkpoints = None
    # Load checkpoints if exist
    if checkpoints != None:
        model.load_state_dict(torch.load(checkpoints, map_location = torch.device("cpu")))
    
    # Set the model, which is a LightningModule
    model = TrainerModel(config, model)

    # Define path to save best model
    best_ckpts_path = os.path.join(results_path, "best_checkpoints")

    plt = pl.Trainer(max_steps = args.max_steps, val_check_interval=args.val_check_interval,
                     check_val_every_n_epoch=None, devices=1, accelerator = 'auto', 
                     strategy = 'ddp_find_unused_parameters_true',
                     enable_checkpointing = False)

    plt.fit(model, train_dataloaders = train_loader, val_dataloaders = val_loader)

    del train_dataset
    del valid_dataset
    del train_loader
    del val_loader

    if test_available:
        plt.test(dataloaders = test_loader, ckpt_path = model.best_ckpts_path)
        del test_dataset
        del test_loader


if __name__ == "__main__":
    
    from egn import EGN

    Experim_Name = args.exp_name
    wandb.init(project='spared_egn_sota', config=vars(args), name=Experim_Name)
    main(dataset=args.dataset, data_batch_size=args.batch_size, learning_r=args.lr, args=args)
    wandb.finish()