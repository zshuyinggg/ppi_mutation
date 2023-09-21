
import torch
from torch_geometric.data import Data
global top_path,script_path, data_path, logging_path
import os, sys
from torch_geometric.loader import DataLoader, NodeLoader
import lightning.pytorch as pl
import torch.utils.data as data
from torch_geometric.data import Dataset
import pandas as pd
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_sparse import SparseTensor

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
        self.wild_embedding_path=wild_embedding_path
        self.wild_embeddings=torch.load(self.wild_embedding_path)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.dic_index2wild_embeddings=self.get_all_indexed_wild_embeddings()
        self.get_edge_index()
        with open(pj(self.processed_dir,'variant_name_list.txt'),'r') as f:
            self.variant_name_list=eval(f.readline())
        self.num_nodes=len(self.dic_index2wild_embeddings)

    @property
    def processed_file_names(self):
        return os.listdir(self.processed_dir)


    def process(self):
        # read variant
        self.wild_embeddings=torch.load(self.wild_embedding_path)
        dic_index2wild_embeddings=self.get_all_indexed_wild_embeddings()
        num_nodes=len(dic_index2wild_embeddings)
        self.get_edge_index()
        self.variant_name_list=self.clinvar_csv['Name'].tolist()
        for name in self.variant_name_list:
            try:
                uniprot=self.variant_embeddings[name]['UniProt']  #check if the variant embedding was not skipped due to drop_last=True
                idx=self.dic_uniprot2idx[uniprot]
            except KeyError:
                self.variant_name_list.remove(name)

        with open(pj(self.processed_dir,'variant_name_list.txt'),'w') as f:
            f.writelines(str(self.variant_name_list))
            print('Processed variant name list')

        
    def len(self):
        return len(self.variant_name_list)

    def get(self, idx):
        name=self.variant_name_list[idx]

        variant_embeddings,variant_uniprot,variant_label=self.variant_embeddings[name]['embs'],self.variant_embeddings[name]['UniProt'],self.variant_embeddings[name]['label']
        index_variant=self.dic_uniprot2idx[variant_uniprot]
        ppi_with_variant_embeddings=torch.vstack([self.dic_index2wild_embeddings[i] if i!=index_variant else variant_embeddings for i in range(len(self.dic_index2wild_embeddings))])
        data = Data(x=(ppi_with_variant_embeddings,variant_embeddings,index_variant),edge_index=self.edge_index,y=[variant_label],num_nodes=self.num_nodes)
        # data = Data(x=(ppi_with_variant_embeddings,variant_embeddings,index_variant),edge_index=self.adj,y=[variant_label],num_nodes=self.num_nodes)
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
        indexed_interactions_edges=[[dic_uniprot2idx[item.split('-')[0]],dic_uniprot2idx[item.split('-')[1]]] for item in exclude_self]
        indexed_interactions_edges_reverse=[[dic_uniprot2idx[item.split('-')[0]],dic_uniprot2idx[item.split('-')[1]]] for item in exclude_self]
        edge_index=torch.tensor(indexed_interactions_edges+indexed_interactions_edges_reverse,dtype=torch.long)
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
    


class VariantRandomWalkLoader(NodeLoader):
    def __init__(data, node_sampler, input_nodes,filter_per_worker=True):
        
        pass

class VariantRandomWalkSampler(GraphSAINTRandomWalkSampler):
    def __init__(self, data, batch_size: int, walk_length: int,
                 num_steps: int = 1, sample_coverage: int = 0,
                 save_dir=None, log: bool = True, **kwargs):
        self.walk_length = walk_length
        data.num_nodes=data.Dataset[0].num_nodes
        data.num_edges=data.Dataset[0].num_edges
        data.edge_index=data.Dataset[0].edge_index
        super().__init__(data, batch_size, num_steps, sample_coverage,
                         save_dir, log, **kwargs)
        self.variant_idx=data.x[2]
    def _sample_nodes(self, batch_size):
        start=torch.tensor(self.variant_idx,dtype=torch.long).reshape((batch_size,))
        # start = torch.randint(0, self.N, (batch_size, ), dtype=torch.long)
        node_idx = self.adj.random_walk(start.flatten(), self.walk_length)
        return node_idx.view(-1)