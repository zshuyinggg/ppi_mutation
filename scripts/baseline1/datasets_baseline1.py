
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
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
from lightning.pytorch import seed_everything

def find_current_path():
    if getattr(sys, 'frozen', False):current = sys.executable
    else:current = __file__
    return current
seed_everything(1050, workers=True)
pj=os.path.join
top_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(find_current_path()))))
sys.path.append(top_path)
















class VariantPPI(Dataset):
    def __init__(self, root, variant_embedding_path, wild_embedding_path, transform=None, pre_transform=None,variant_list_name='2019_variant_name_list', pre_filter=None, **args):
        self.variant_embeddings=torch.load(variant_embedding_path)
        self.wild_embedding_path=wild_embedding_path
        self.wild_embeddings=torch.load(self.wild_embedding_path)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.dic_index2wild_embeddings=self.get_all_indexed_wild_embeddings()
        self.get_edge_index()
        self.variant_name_list=read_variant_name_list_screen(variant_embedding_path,variant_list_name,self.dic_uniprot2idx)
        self.num_nodes=len(self.dic_index2wild_embeddings)

    @property
    def processed_file_names(self):
        return os.listdir(self.processed_dir)


    # def process(self,variant_only=False):
    #     # read variant
    #     if not variant_only:
    #         self.wild_embeddings=torch.load(self.wild_embedding_path)
    #         dic_index2wild_embeddings=self.get_all_indexed_wild_embeddings()
    #         num_nodes=len(dic_index2wild_embeddings)
    #         self.get_edge_index()
    #     print('processing')
    #     self.variant_name_list=self.clinvar_csv['Name'].tolist()
    #     for name in self.variant_name_list:
    #         try:
    #             uniprot=self.variant_embeddings[name]['UniProt']  #check if the variant embedding was not skipped due to drop_last=True
    #             idx=self.dic_uniprot2idx[uniprot]
    #         except KeyError:
    #             self.variant_name_list.remove(name)

    #     with open(pj(self.processed_dir,'%s.txt'%self.variant_list_name),'w') as f:
    #         f.writelines(str(self.variant_name_list))
    #         print('Processed variant name list')

        
    def len(self):
        return len(self.variant_name_list)

    def get(self, idx):
        name=self.variant_name_list[idx]

        try:variant_embeddings,variant_uniprot,variant_label=self.variant_embeddings[name]['embs'],self.variant_embeddings[name]['UniProt'],self.variant_embeddings[name]['label']
        except:
            print(name,flush=True)
        
        index_variant=self.dic_uniprot2idx[variant_uniprot]
        ppi_with_variant_embeddings=torch.vstack([self.dic_index2wild_embeddings[i] if i!=index_variant else variant_embeddings for i in range(len(self.dic_index2wild_embeddings))])
        data = Data(x=(ppi_with_variant_embeddings,variant_embeddings,index_variant),edge_index=self.edge_index,y=[variant_label],num_nodes=self.num_nodes)
        # data = Data(x=(ppi_with_variant_embeddings,variant_embeddings,index_variant),edge_index=self.adj,y=[variant_label],num_nodes=self.num_nodes)
        return data

    def get_stat(self):
        l=[]
        for idx in range(self.__len__()):
            label=self.variant_embeddings[self.variant_name_list[idx]]['label']
            l.append(label)
        print('overall data num=%s, positive samples= %s'%(len(l),sum(l)))
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
    def __init__(self,root, train_val_variant_embedding_path, wild_embedding_path,batch_size,num_workers,\
                 random_seed,test_variant_embedding_path=None,\
                    train_list_name='2019_train_name_list_1050',\
                        val_list_name='2019_val_name_list_1050',\
                            test_list_name='2019_test_name_list_1050',\
                                **args) :
        super().__init__()
        self.train_set=VariantPPI(root, train_val_variant_embedding_path, wild_embedding_path,variant_list_name=train_list_name)
        self.val_set=VariantPPI(root, train_val_variant_embedding_path, wild_embedding_path,variant_list_name=val_list_name)
        
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.random_seed=random_seed
        # self.trainset,self.valset=split_train_val(self.dataset,train_val_split=train_val_ratio,random_seed=random_seed)
        self.test_set=VariantPPI(root, test_variant_embedding_path, wild_embedding_path,variant_list_name=test_list_name)
        self.train_set.get_stat()
        self.val_set.get_stat()
        self.test_set.get_stat()
    def train_dataloader(self) :
        return DataLoader(self.train_set,shuffle=True,batch_size=self.batch_size,num_workers=self.num_workers)
    
    def val_dataloader(self) :
        return DataLoader(self.val_set,shuffle=False,batch_size=2*self.batch_size,num_workers=self.num_workers)
    
    def test_dataloader(self) :
        return DataLoader(self.test_set,shuffle=False,batch_size=2*self.batch_size,num_workers=self.num_workers)

class VariantRandomWalkLoader(NodeLoader):
    def __init__(data, node_sampler, input_nodes,filter_per_worker=True):
        
        
        if node_sampler is None:
            neighbor_sampler = VariantRandomWalkSampler(
                data,
                num_neighbors=num_neighbors,
                replace=replace,
                directed=directed,
                disjoint=disjoint,
                temporal_strategy=temporal_strategy,
                time_attr=time_attr,
                is_sorted=is_sorted,
                share_memory=kwargs.get('num_workers', 0) > 0,
            )
        super().__init__(
            data=data,
            node_sampler=neighbor_sampler,
            input_nodes=input_nodes,
            input_time=input_time,
            transform=transform,
            transform_sampler_output=transform_sampler_output,
            filter_per_worker=filter_per_worker,
            **kwargs,
        )

        
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
    

def read_variant_name_list_screen(emb_path,variant_list_name,uniprotdic):
    embs=torch.load(emb_path)
    pth=pj('/scratch/user/zshuying/ppi_mutation/data/baseline1/processed','%s.txt'%variant_list_name)
    with open(pth,'r') as f:
        variant_name_list=eval(f.readline())
    # for name in variant_name_list:
    #     if embs.get(name):
    #         if (embs[name].get('embs') is not None and embs[name].get('UniProt')):pass
    #         else:
    #             variant_name_list.remove(name) 
    #             print('removing %s'%name)
    #     else:
    #         variant_name_list.remove(name) 
    #         print('removing %s'%name)
    # with open(pth,'w') as f:
    #     f.writelines(str(variant_name_list))
    # print('%s saved'%pth)
    return variant_name_list


def variant_list_from_embs(emb_path,variant_list_name):
    embs=torch.load(emb_path)
    pth=pj('/scratch/user/zshuying/ppi_mutation/data/baseline1/processed','%s.txt'%variant_list_name)
    variant_name_list=[]
    for name in embs:
        if (embs[name].get('embs') is not None and embs[name].get('UniProt')):variant_name_list.append(name)
        else:
            pass
    with open(pth,'w') as f:
        f.writelines(str(variant_name_list))
    print('%s saved'%pth)
    return variant_name_list


def set_variant_name_list_ratio(variant_list_name,variant_embedding_path,ratio,save_name):
    #ratio is the pos:neg ratio
    pth=pj('/scratch/user/zshuying/ppi_mutation/data/baseline1/processed','%s.txt'%variant_list_name)
    with open(pth,'r') as f:
        variant_name_list=eval(f.readline())
    embs=torch.load(variant_embedding_path)
    names=[]
    labels=[]
    for name in embs:
        names.append(name)
        labels.append(embs[name]['label'])
    pos_neg_ratio_ori=sum(labels)/(len(labels)-sum(labels))
    if pos_neg_ratio_ori>ratio:
        neg=len(labels)-sum(labels)
        pos=int(neg * ratio)
    else:
        pos=sum(labels)
        neg=pos//ratio

    df=pd.DataFrame(list(zip(names,labels)),columns=['Name','Label'])
    pos_names=df[df['Label']==1].sample(n=int(pos))['Name'].tolist()
    neg_names=df[df['Label']==0].sample(n=int(neg))['Name'].tolist()
    print('positive samples: %d, negative samples: %d'%(len(pos_names),len(neg_names)))
    with open(pj('/scratch/user/zshuying/ppi_mutation/data/baseline1/processed','%s.txt'%save_name),'w') as f:
        f.writelines(str(pos_names+neg_names))
    
# variant_list_from_embs('/scratch/user/zshuying/ppi_mutation/data/baseline0/2019_test_variant_embds.pt','2019_test_variant_name_list')


# set_variant_name_list_ratio('2019_test_name_list_1050','/scratch/user/zshuying/ppi_mutation/data/baseline0/2019_test_variant_embds.pt',1.8,'2019_test_1.8_variant_name_list')