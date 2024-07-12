import os
import torch
import heapq
import torchvision
import numpy as np
from joblib import Parallel, delayed
from torchvision import transforms
import torch.nn.functional as F

# Using spare library
from spared.datasets import get_dataset
from utils import *
parser = get_main_parser()
args = parser.parse_args()
use_cuda = torch.cuda.is_available()

# Declare device
device = torch.device("cuda" if use_cuda else "cpu")

# Get dataset from the values defined in args
dataset = get_dataset(args.dataset)

# Declare train and test datasets
train_dataloader = dataset.adata[dataset.adata.obs['split']=='train']
val_dataloader = dataset.adata[dataset.adata.obs['split']=='val']
test_dataloader = dataset.adata[dataset.adata.obs['split']=='test']

names = ['train', 'val', 'test']
test_available = False if test_dataloader.shape[0] == 0 else True
loaders = [train_dataloader, val_dataloader, test_dataloader] if test_available else [train_dataloader, val_dataloader]

# Define transformations for the patches
train_transforms = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

save_dir = os.path.join('EGN_exemplars', args.dataset)
os.makedirs(save_dir, exist_ok=True)

TORCH_HOME = '~/.torch'

os.environ['TORCH_HOME'] = TORCH_HOME
encoder = torchvision.models.resnet50(True)
features = encoder.fc.in_features
modules=list(encoder.children())[:-1] # Encoder corresponds to ResNet50 without the fc layer
encoder=torch.nn.Sequential(*modules)
for p in encoder.parameters():
    p.requires_grad = False

encoder = encoder.to(device)
encoder.eval()

num_cores = 12
   
            
def generate():
   
    def extract(patches):
        '''
        This function receives the 2D patches and feeds them to the encoder.
        '''
        
        return encoder(patches).view(-1,features) # shape post encoder [64, 2048, 1, 1] -> turns to [64, 2048]
    
    #Iterate over the two datasets (Train and Test)
    for dir in range(len(loaders)):
        # img_embedding will save the embedding representation (size 2048) of each one of the patches in loaders[dir]
        img_embedding = []

        #Iterate over all the patches in the train or test loader
        for data in loaders[dir]:
            # Get patches of the whole slide image contained in dataloader and unflatten them
            tissue_tiles = torch.tensor(data.obsm['patches_scale_1.0'])
            tissue_tiles = tissue_tiles.reshape((tissue_tiles.shape[0], round(np.sqrt(tissue_tiles.shape[1]/3)), round(np.sqrt(tissue_tiles.shape[1]/3)), -1))
        
            # Permute dimensions to be in correct order for normalization
            tissue_tiles = tissue_tiles.permute(0,3,1,2).contiguous()
            # Resize patches
            tissue_tiles = F.interpolate(tissue_tiles, size=(256, 256), mode='bilinear', align_corners=False)
        
            # Make transformations in tissue tiles
            tissue_tiles = train_transforms(tissue_tiles/255.)
            
            # extract patch encoding
            img_embedding += [extract(tissue_tiles.cuda())]

        # Concatenate the embedding of all the patches of the loaded image
        img_embedding = torch.cat(img_embedding).contiguous()
        print(img_embedding.size())

        # Save tensor of shape [#patches, 2048]. This tensor corresponds to the whole image encoded (or the entire split).
        torch.save(img_embedding, f"{save_dir}/{names[dir]}.pt")
    

def create_search_index():
    # Make dir where the KNNs for the dataloader image will be saved
    os.makedirs(os.path.join(save_dir, 'knns_search'), exist_ok=True)
    
    class Queue:
        def __init__(self, max_size = 2):
            self.max_size = max_size
            self.list = []

        def add(self, item):

            heapq.heappush(self.list, item)

            while len(self.list) > self.max_size:
                heapq.heappop(self.list)
    
        def __repr__(self):
            return str(self.list)

    # Load encoded representation of train and test patches
    train_encode = f"{save_dir}/{names[0]}.pt"
    train_encode = torch.load(train_encode).cuda()

    val_encode = f"{save_dir}/{names[1]}.pt"
    val_encode = torch.load(val_encode).cuda()

    if test_available:
        test_encode = f"{save_dir}/{names[2]}.pt"
        test_encode = torch.load(test_encode).cuda()

    encodes = [train_encode, val_encode, test_encode] if test_available else [train_encode, val_encode]

    for i in range(len(encodes)):
        print(f"Finding the 100 neirest neigbors for each patch in {names[i]}")
        # train_Q will later save the KNNs for each patch. These neighbors come from the train image
        Q = [Queue(max_size=100) for _ in range(encodes[i].size(0))]

        # Calculate the distance between the embedding representation of the "main" patches and the train patches 
        dist = torch.cdist(encodes[i].unsqueeze(0), train_encode.unsqueeze(0), p = 2).squeeze(0)

        if i == 0:
            # If we are getting the KNNs of the train dataset, ignore the closest NN (it is the patch itself)
            topk = min(len(dist), 101)

            # Take the top 100 distances and save their value and index
            knn = dist.topk(topk, dim = 1, largest=False) 
            q_values = knn.values[:, 1:].cpu().numpy()
            q_infos =  knn.indices[:, 1:].cpu().numpy()          
        
        else:
            topk = min(len(dist), 100)
            knn = dist.topk(topk, dim = 1, largest=False)
            q_values = knn.values.cpu().numpy()
            q_infos =  knn.indices.cpu().numpy() 

        def add(q_value,q_info, myQ):
            # This function adds the neighbor's distance to the main patch and the neighbor's index to the list of KNNs
            for idx in range(len(q_value)):
                myQ.add((-q_value[idx],q_info[idx]))
            return myQ

        Q = Parallel(n_jobs=num_cores)(delayed(add)(q_values[f],q_infos[f],Q[f]) for f in range(q_values.shape[0]))

        # Save the KNN file
        np.save(f"{save_dir}/knns_search/{names[i]}.npy", [myq.list for myq in Q]) 


if __name__=='__main__':
    # Check if the exemplars have already been created.
    if not os.path.exists(os.path.join(save_dir, 'train.pt')):
        print("Creating exemplars")
        generate()
        print(f"Exemplars saved in {save_dir}")

    # Check if the nearest neighbors have already been searched for.
    if not os.path.exists(os.path.join(save_dir, 'knns_search', 'train.npy')):
        print("Finding KNNs for each patch") 
        create_search_index()
        print(f"KNNs retrieved and saved in {save_dir}")