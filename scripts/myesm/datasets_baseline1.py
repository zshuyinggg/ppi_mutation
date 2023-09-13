
import torch
from torch_geometric.data import Data
global top_path,script_path, data_path, logging_path
import os, sys
from torch_geometric.loader import DataLoader
import lightning.pytorch as pl
import torch.utils.data as data
from torch_geometric.data import Dataset
import pandas as pd

def find_current_path():
    if getattr(sys, 'frozen', False):current = sys.executable
    else:current = __file__
    return current

pj=os.path.join
top_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(find_current_path()))))
sys.path.append(top_path)


class VariantPPI(Dataset):
    def __init__(self, root, clinvar_csv, variant_embedding_path, wild_embedding_path, transform=None, pre_transform=None, pre_filter=None):
        self.clinvar_csv=pd.read_csv(clinvar_csv)
        self.variant_embeddings=torch.load(variant_embedding_path)
        self.wild_embeddings=torch.load(wild_embedding_path)
        super().__init__(root, transform, pre_transform, pre_filter)


    @property
    def processed_file_names(self):
        return os.listdir(self.processed_dir)


    def process(self):
        # read variant
        dic_index2wild_embeddings=self.get_all_indexed_wild_embeddings()
        num_nodes=len(dic_index2wild_embeddings)
        self.get_edge_index()
        idx = 0
        for name in self.clinvar_csv['Name'].tolist():
            if f'data_{idx}.pt' in os.listdir(self.processed_dir):
                idx+=1
                continue
            try:
                variant_embeddings,variant_uniprot,variant_label=self.variant_embeddings[name]['embs'],self.variant_embeddings[name]['UniProt'],self.variant_embeddings[name]['label']
                index_variant=self.dic_uniprot2idx[variant_uniprot]

                ppi_with_variant_embeddings=torch.vstack([dic_index2wild_embeddings[i] if i!=index_variant else variant_embeddings for i in range(len(dic_index2wild_embeddings))])
                data = Data(x=(ppi_with_variant_embeddings,variant_embeddings,index_variant),edge_index=self.edge_index,y=[variant_label],num_nodes=num_nodes)
                torch.save(data, pj(self.processed_dir, f'data_{idx}.pt'))
                print(f'data_{idx}.pt saved')
                idx += 1
            except KeyError:
                print('the embedding of %s is not found. Probably due to the drop_last=True of dataloader in embeddings_from_baseline0.py. Skipping.'%name)
                continue
        
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(pj(self.processed_dir, f'data_{idx}.pt'))
        return data

    def get_all_indexed_wild_embeddings(self):
        uniprots,dic_uniprot2idx,dic_idx2uniprot=all_uniprot()
        self.dic_uniprot2idx,self.dic_idx2uniprot=dic_uniprot2idx,dic_idx2uniprot
        dic_index2wild_embeddings=dict([(dic_uniprot2idx[uniprot],self.wild_embeddings[uniprot]['embs']) for uniprot in uniprots])
        return dic_index2wild_embeddings
    
    def get_edge_index(self):
        uniprots,dic_uniprot2idx,dic_idx2uniprot=all_uniprot()
        self.dic_uniprot2idx,self.dic_idx2uniprot=dic_uniprot2idx,dic_idx2uniprot
        with open(pj(top_path,'ppi_huri_humap.txt'),'r') as f:
            interactions=eval(f.readline())
        exclude_self=[item for item in interactions if '-' in item]
        indexed_interactions_eges=[[dic_uniprot2idx[item.split('-')[0]],dic_uniprot2idx[item.split('-')[1]]] for item in exclude_self]
        indexed_interactions_eges_reverse=[[dic_uniprot2idx[item.split('-')[0]],dic_uniprot2idx[item.split('-')[1]]] for item in exclude_self]
        edge_index=torch.tensor(indexed_interactions_eges+indexed_interactions_eges_reverse,dtype=torch.long)
        self.edge_index=edge_index.t().contiguous()
        return


def all_uniprot(f_path=pj(top_path,'ppi_seq_huri_humap.csv')):
    uniprots=pd.read_csv(f_path)['UniProt'].unique().tolist()
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