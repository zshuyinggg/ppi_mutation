#%%
top_path = '/scratch/user/zshuying/ppi_mutation/scripts'

import torch
from sklearn.decomposition import PCA

from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import Dataset
global top_path  # the path of the top_level directory
global script_path, data_path, logging_path
import os, sys
sys.path.append(top_path)
from torch.utils.data import DataLoader
import esm
from baseline0.datasets import *
import numpy as np
from argparse import ArgumentParser
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from baseline0.model import *
import pandas as pd
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt

import pandas as pd
test_set=pd.read_csv('/scratch/user/zshuying/ppi_mutation/data/clinvar/mutant_seq_2019_test_no_error.csv'
)
with open('/scratch/user/zshuying/ppi_mutation/data/baseline1/processed/2019_test_name_list_1050.txt','r') as f:
    test_list=eval(f.readline())
with open('/scratch/user/zshuying/ppi_mutation/data/baseline1/processed/2019_train_name_list_1050.txt','r') as f:
    train_list=eval(f.readline())
with open('/scratch/user/zshuying/ppi_mutation/data/baseline1/processed/2019_val_name_list_1050.txt','r') as f:
    val_list=eval(f.readline())
train_val=pd.read_csv('/scratch/user/zshuying/ppi_mutation/data/clinvar/mutant_seq_2019_1_no_error.csv')
test_actual_df=test_set[test_set['Name'].apply(lambda x:x in test_list)]
test_uniprot=set(test_actual_df.UniProt.unique().tolist())
train_actual_df=train_val[train_val['Name'].apply(lambda x:x in train_list)]
train_uniprot=set(train_actual_df.UniProt.unique().tolist())
val_actual_df=train_val[train_val['Name'].apply(lambda x:x in val_list)]
val_uniprot=set(val_actual_df.UniProt.unique().tolist())
pth=torch.load('/scratch/user/zshuying                                     /ppi_mutation/data/baseline1/all_wild_esm_embds.pt')
seqs=[]
y=[]
uniprots=pd.read_csv('/scratch/user/zshuying/ppi_mutation/ppi_seq_huri_humap.csv')['UniProt'].unique().tolist()
for uniprot in uniprots:
    seqs.append(pth[uniprot]['embs'])
    if uniprot in train_uniprot:
        y.append(0)
    elif uniprot in val_uniprot:
        y.append(1)
    elif uniprot in test_uniprot:
        y.append(2)
    else:
        y.append(3)
seqs_data=np.array(seqs)


#%%
umap_reducer=umap.UMAP(n_neighbors=20)
embedding_reduced = umap_reducer.fit_transform(seqs)

#%%
cdict = {0: ('green',0),1: ('orange',1), 2: ('blue',0.5), 3: ('green',0)}
colored_y=[cdict[y] for y in y]
marker_size_dict=[3,8,3,3]
marker_y=[marker_size_dict[y] for y in y]
fig_dims = (10, 10)
fig, ax = plt.subplots(figsize=fig_dims)
sc = ax.scatter(embedding_reduced[:, 0], embedding_reduced[:, 1], s= marker_y,c=colored_y)
plt.title('Embedding of the wild proteins from ESM by UMAP');
markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in cdict.values()]
plt.legend(markers, ['train_wild','val_wild','test_wild','ppi_wild not variant'], numpoints=1,fontsize=16)
