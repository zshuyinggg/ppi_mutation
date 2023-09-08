
import torch
from torch_geometric.data import Data
global top_path  # the path of the top_level directory
global script_path, data_path, logging_path
import os, sys
from torch.utils.data import DataLoader

import lightning.pytorch as pl

def find_current_path():
    if getattr(sys, 'frozen', False):
        # The application is frozen
        current = sys.executable
    else:
        # The application is not frozen
        # Change this bit to match where you store your data files:
        current = __file__

    return current

pj=os.path.join
top_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(find_current_path()))))
sys.path.append(top_path)

import pandas as pd
from torch_geometric.data import Dataset

import torch.utils.data as data

class VariantPPI(Dataset):
    def __init__(self, root, clinvar_csv, variant_embedding_path, wild_embedding_path, transform=None, pre_transform=None, pre_filter=None):
        self.clinvar_csv=pd.read_csv(clinvar_csv)
        self.variant_embeddings=torch.load(variant_embedding_path)
        self.wild_embeddings=torch.load(wild_embedding_path)
        super().__init__(root, transform, pre_transform, pre_filter)


    @property
    def processed_file_names(self):
        return [f'data_{idx}.pt' for idx in range(len(self.clinvar_csv))]


    def process(self):
        # read variant
        dic_index2wild_embeddings=self.get_all_indexed_wild_embeddings()
        idx = 0
        for name in self.clinvar_csv['Name'].tolist():
            variant_embeddings,variant_uniprot,variant_label=self.variant_embeddings[name]['embs'],self.variant_embeddings[name]['UniProt'],self.variant_embeddings[name]['label']
            index_variant=self.dic_uniprot2idx(variant_uniprot)

            ppi_with_variant_embeddings=torch.vstack([dic_index2wild_embeddings[i] if i!=index_variant else variant_embeddings for i in range(len(dic_index2wild_embeddings))])
            data = Data(x=(ppi_with_variant_embeddings,variant_embeddings,index_variant),edge_index=self.edge_index,y=[variant_label])
            torch.save(data, pj(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(pj(self.processed_dir, f'data_{idx}.pt'))
        return data

    def get_all_indexed_wild_embeddings(self):
        uniprots,dic_uniprot2idx,dic_idx2uniprot=all_clinvar_uniprot(self.clinvar_csv)
        self.dic_uniprot2idx,self.dic_idx2uniprot=dic_uniprot2idx,dic_idx2uniprot
        dic_index2wild_embeddings=dict([(dic_uniprot2idx[uniprot],self.wild_embeddings[uniprot]) for uniprot in uniprots])
        return dic_index2wild_embeddings
    
    def get_indexed_ppi(self):
        uniprots,dic_uniprot2idx,dic_idx2uniprot=all_clinvar_uniprot(self.clinvar_csv)
        self.dic_uniprot2idx,self.dic_idx2uniprot=dic_uniprot2idx,dic_idx2uniprot
        with open('/scratch/user/zshuying/ppi_mutation/data/ppi_huri_humap.txt','r') as f:
            interactions=eval(f.readline())
        exclude_self=[item for item in interactions if '-' in item]
        indexed_interactions_eges=[[dic_uniprot2idx[item.split('-')[0]],dic_uniprot2idx[item.split('-')[1]]] for item in exclude_self]
        indexed_interactions_eges_reverse=[[dic_uniprot2idx[item.split('-')[0]],dic_uniprot2idx[item.split('-')[1]]] for item in exclude_self]
        edge_index=torch.tensor(indexed_interactions_eges+indexed_interactions_eges_reverse,dtype=torch.long)
        self.edge_index=edge_index.t().contiguous()
        return


def all_clinvar_uniprot(df):
    uniprots=df['UniProt'].unique().tolist()
    dic_uniprot2idx=dict([(uniprots[i],i) for i in range(len(uniprots))])
    dic_idx2uniprot=dict([(i,uniprots[i]) for i in range(len(uniprots))])
    return uniprots, dic_uniprot2idx,dic_idx2uniprot


def split_train_val(dataset,train_val_split=0.8,random_seed=52):
    train_set_size=int(len(dataset)*train_val_split)
    valid_set_size=len(dataset)-train_set_size
    seed=torch.Generator().manual_seed(random_seed)
    train_set,valid_set=data.random_split(dataset,[train_set_size,valid_set_size],generator=seed)
    print('Split dataset into train, val with the rate of %s'%train_val_split)
    return train_set,valid_set


class VariantPPIModule(pl.LightningDataModule):
    def __init__(self,root, clinvar_csv, variant_embedding_path, wild_embedding_path,batch_size,num_workers,random_seed,train_val_ratio=0.8) :
        super().__init__()
        self.dataset=VariantPPI(root, clinvar_csv, variant_embedding_path, wild_embedding_path)
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.random_seed=random_seed
        self.trainset,self.valset=split_train_val(self.dataset,train_val_split=train_val_ratio,random_seed=random_seed)
    def train_dataloader(self) :
        return DataLoader(self.trainset,shuffle=True,batch_size=self.batch_size,num_workers=self.num_workers)
    
    def val_dataloader(self) :
        return DataLoader(self.valset,shuffle=False,batch_size=self.batch_size,num_workers=self.num_workers)