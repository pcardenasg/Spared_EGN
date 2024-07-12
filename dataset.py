from torch.utils.data import Dataset
import numpy as np
import torch
import os
import pickle
import pandas
import tifffile
from PIL import Image
from scanpy import read_visium 
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import torch.nn.functional as F

'''
This file organizes the Spatial Transcriptomics SPARED pre-processed dataset to the format in which EGN expects it from the dataloader.
Each element in the Dataloader corresponds to a dictionary with the patch tensor [3, 256, 256], the counts vector and
the embedding representation of the main patch, the counts vector and embedding representations of the numk (6) nearest
neighbors, and the coordinates of each patch in the WSI.
'''

class OrganizeDataset(Dataset):
    def __init__(self, dataset, args, data_transforms, exemplar_path, split_name, pred_layer, op_data = None): # Examples -> dataset = train_dataset; exemplar_path = "Dataset_ST"
        # if processing test dataset, op_data must be train dataset
        self.all_patches_data = dataset # dataset = train_dataset
        self.transforms = data_transforms
        self.exemplar_path = exemplar_path
        self.numk = args.numk
        self.split_name = split_name
        self.pred_layer = pred_layer
        self.all_counts = torch.tensor(self.all_patches_data.layers[self.pred_layer])
        # FIXME: revisar resultados de predicción sobre c_t_log1p sin enviar la máscara "mask"
        self.all_masks = torch.tensor(self.all_patches_data.layers["mask"])

        # Load the latent representation of all the train/test patches
        self.image_encode = torch.load(f"{self.exemplar_path}/{self.split_name}.pt", map_location=torch.device("cpu"))
        
        if split_name != 'train':
            # Load the latent representation of the "Train"/opposite patches for retrieving the neirest neighbors' data
            self.op_encode = torch.load(f"{self.exemplar_path}/train.pt", map_location=torch.device("cpu"))
            
            # Load the KNN patches for latter visualization
            self.op_patches = op_data.obsm['patches_scale_1.0']

            # Load the gene expression counts of the "Train"/opoosite patches for retrieving the neirest neighbors' data
            self.op_expressions = torch.tensor(op_data.layers[self.pred_layer])
            self.op_masks = torch.tensor(op_data.layers["mask"])
        
        else:
            # Same data as previous condition, but for the train dataset as the "main" data. - Opposite patches only come from the "Train" set.
            self.op_encode = self.image_encode.clone()
            self.op_patches = self.all_patches_data.obsm['patches_scale_1.0']
            self.op_expressions = torch.tensor(self.all_patches_data.layers[self.pred_layer])
            self.op_masks = torch.tensor(self.all_patches_data.layers["mask"])

        # Load the KNNs distances and idxs for the "main" data
        self.knns = np.load(os.path.join(self.exemplar_path, f"knns_search/{self.split_name}.npy"))

    def retrive_similer(self, index):
        # Default numk from original EGN repository = 6 
        # Organizes the 100 NN in distance order (from furthest to nearest) in order to obtain the (numk) closest NN.
        index = np.array([[i,j] for i,j in sorted(index, key = lambda x : float(x[0]))]) 
        index = index[-self.numk:]
    
        # op_emb (op_features in the final dictionary) - Ends as a list of the embeddings of the (numk) closest patches to the main patch.
        op_emb = []
        # op_counts (op_count in the final dictionary) - Ends as a list of the expression counts of the 1024 genes in the (numk) closest patches to the main patch.
        op_counts = []
        # iterates through each one of the (numk) closest patches to retreive their individual embedding and counts vectors.
        for _, op_name in index:
            op_name = int(op_name)
            # op_emb is the encoding of the NN 
            op_emb.append(self.op_encode[op_name])
            op_count = self.op_expressions[op_name]
            op_counts.append(op_count)

        # Stack the embeddings and counts of the (numk) NN of one patch of interest.
        return torch.stack(op_emb).view(self.numk,-1), torch.stack(op_counts).view(self.numk, op_count.size()[0]).to(torch.float32)

    def patch_dict(self, idx, patch_emb):
        ''' 
        This function creates the dictionary that the model receives from the Dataloader
        '''
        # Get the Ann Data of the patch "idx" and its corresponding 100 NN.
        patch_data = self.all_patches_data[idx]
        coord = patch_data.obsm['spatial'][0]
        knn_for_patch = self.knns[idx]

        # Find and pre-process the visual data of the patch "idx".
        patch = torch.tensor(patch_data.obsm['patches_scale_1.0'][0])
        patch = patch.reshape((round(np.sqrt(patch.shape[0]/3)), round(np.sqrt(patch.shape[0]/3)), -1))
        patch = patch.permute(2, 0, 1).contiguous()
        
        # Resize patch to 256 (EGN image size)
        patch = patch.unsqueeze(0)
        patch = F.interpolate(patch, size=(256, 256), mode='bilinear', align_corners=False)
        patch = patch.squeeze(0)

        patch = self.transforms(patch/255.)
        # Find the ground truth gene expression for patch "idx".
        counts = self.all_counts[idx]

        # Retrieve the embedding and counts vectors of the closest NN
        op_emb, op_counts = self.retrive_similer(knn_for_patch)
        # Find the coordinates of the patch "idx" in the WSI.
        pos = torch.tensor((self.all_patches_data.obs['array_col'][idx], self.all_patches_data.obs['array_row'][idx]))

        # new key "mask" for masking the missing counts in test
        missing_genes_mask = self.all_masks[idx]

        return {
            "img" : patch,
             "count" : counts.to(torch.float32),
             "p_feature": patch_emb,
             "op_count": op_counts,
             "op_feature":op_emb,
             "pos": torch.LongTensor(pos).to(torch.float32),
             # new key "mask" for masking the missing counts in test
             "mask": missing_genes_mask
            }
    
    def __getitem__(self, index):

        # Get the embedding of patch "idx" and create the data dictionary.
        patch_emb = self.image_encode[index].unsqueeze(0)
        return self.patch_dict(index, patch_emb)
    
    def __len__(self):
        return len(self.all_patches_data)
    
    def get_patches(self, main_patch_idx, numk):
        '''
        This function returns the numk NN of a "main patch" as a list of tuples with the structure (NN_distance_to_main_patch, NN_patch_tensor).
        No transformations are made since this is only for visualization purposes.
        '''
        # Obtain the tensor of the main patch and process it - unflatten
        patch = torch.tensor(self.all_patches_data.obsm['patches_scale_1.0'][main_patch_idx])
        patch = patch.reshape((round(np.sqrt(patch.shape[0]/3)), round(np.sqrt(patch.shape[0]/3)), -1))
        patch = patch.permute(2, 0, 1).contiguous()
        patches = [('Main patch', patch)]

        # Get the NN of the main patch and sort them from closest to furthest to obtain the numk closest ones (this gets the distances and the idx of the NN to then retrieve their respective tensor).
        index = self.knns[main_patch_idx]
        index = np.array([[i,j] for i,j in sorted(index, key = lambda x : float(x[0]), reverse=True)]) 
        index = index[:numk]
    
        knn_idxs = []
        for dist, op_name in index:
            knn_idxs.append((str(round(dist, 2)), int(op_name)))

        for dist, patch_id in knn_idxs:
            patch = torch.tensor(self.op_patches[patch_id])
            patch = patch.reshape((round(np.sqrt(patch.shape[0]/3)), round(np.sqrt(patch.shape[0]/3)), -1))
            patch = patch.permute(2, 0, 1).contiguous()
            patches.append((dist, patch))

        return patches
        
    def plot_neighbors(self, num_patches: int, num_knns:int):
        '''
        This function allows the visualization of num_knns NN of a set fo patches (num_patches).
        The NN are always retrieved from the "Train" dataset.
        '''

        # Get the idxs of a random set of sample patches
        main_patches = random.sample(range(0, len(self.all_patches_data)), num_patches)
        patch_groups = {}

        # For each patch obtain the list of num_knns NN.
        for main in main_patches:
            patch_groups[main] = self.get_patches(main, num_knns)

        # Create plot.
        fig, axes = plt.subplots(nrows=num_patches, ncols=num_knns+1)
        for i, (patch_idx, value) in enumerate(patch_groups.items()):
            for j, (name, tensor) in enumerate(value):
                
                axes[i, j].imshow(tensor.permute(1, 2, 0))
                axes[i, j].axis('off')
                if i == 0 and j == 0:
                    axes[i, j].set_title(f"{name}")
                elif j != 0:
                    axes[i, j].set_title(f"Dist: {name}")

        fig.tight_layout()
        plt.savefig(f'{self.exemplar_path}/{self.split_name}_{str(num_patches)}patches_{str(num_knns)}KNNs.png')
        