import copy
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

import torch
from torch.utils.data import Dataset
import torch.utils.data as data

global top_path  # the path of the top_level directory
global script_path, data_path, logging_path
import os, sys
from torch.utils.data import DataLoader
import esm
from dask.diagnostics import ProgressBar

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


top_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(find_current_path()))))
sys.path.append(top_path)
from scripts.utils_clinvar import *

from scripts.utils import *
import pandas as pd
import dask.dataframe as ddf
import multiprocessing
from torchvision import transforms, utils

# num_partitions = multiprocessing.cpu_count()-4
num_partitions = 28


class ProteinEmbeddings(Dataset):
    def __init__(self,embedding_path,clinvar_csv) -> None:
        super().__init__()
        self.embeddings=torch.load(embedding_path)
        self.all_sequences = pd.read_csv(clinvar_csv)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx, uniprot=None, label=None):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sequences = self.all_sequences.iloc[idx, self.all_sequences.columns.get_loc('Seq')]
        uniprot = self.all_sequences.iloc[idx, self.all_sequences.columns.get_loc('UniProt')]
        name = self.all_sequences.iloc[idx, self.all_sequences.columns.get_loc('Name')]
        labels=self.all_sequences.iloc[idx,self.all_sequences.columns.get_loc('Label')]
        sample={'label':torch.tensor(labels).int(),'UniProt':uniprot,'Name':name,'Loc':get_loc_from_name(name)} #multiple or single?

        return sample





class ProteinSequence(Dataset):
    """
    Generate protein sequences according to provided clinvar_csv to gen_dir,
    label decides only positive samples(1), only negative samples(0), or both (None)
    """

    def __init__(self, clinvar_csv=os.path.join(script_path, 'merged_2019_1.csv'), 
                 test_mode=False,
                 transform=None,
                 random_seed=52,
                 delta=True):
        """
        labels: 2 wild. 0 negative. 1 possitive
        :param clinvar_csv:
        :param gen_file_path:
        :param gen_file:
        :param test_mode:
        :param train_val_ratio: ration for training set
        :param random_seed:
        """
        self.random_seed=random_seed
        print('Reading file')
        self.all_sequences = pd.read_csv(clinvar_csv)
        self.test_mode=test_mode
        self.transform=transform

        if test_mode: 
            self.all_sequences=self.all_sequences.loc[:10,:]
        if test_mode:self.all_sequences=self.all_sequences[:10]
        self.remove_na()
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        self.all_sequences=self.all_sequences.sample(frac=1,random_state=random_seed).reset_index(drop=True)
        #set this to class property to make sure train and val are split on the same indexes
        self.all_sequences.loc[self.all_sequences['Name']=='0','Label']=0 #wild type is considered the same as benign
        self.all_sequences.loc[self.all_sequences['Label']==-1,'Label']=0 
        if delta:
            self.all_sequences=self.all_sequences[self.all_sequences['Name']!='0']
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
            self.all_sequences=self.all_sequences.sample(frac=1,random_state=random_seed).reset_index(drop=True)
        print(self.all_sequences['Label'].describe())
    def get_idx_from_uniprot(self,uniprot):
        idx = self.all_sequences[self.all_sequences['UniProt']==uniprot].index
        return idx
    def remove_na(self):
        self.all_sequences.dropna(inplace=True,ignore_index=True)
        print('nan removed')

    def set_class_seq(self):
        ProteinSequence.all_sequences=self.all_sequences

    def correct_labels(self):

        # self.all_sequences['Label']=(self.all_sequences['Label']+1)/2
        self.all_sequences.loc[self.all_sequences['Name']=='0','Label']=2
        print(self.all_sequences['Label'].describe())
        # self.all_sequences.to_csv(self.gen_file_path)


    


    def __len__(self):
        return len(self.all_sequences)

    def __getitem__(self, idx, uniprot=None, label=None):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sequences = self.all_sequences.iloc[idx, self.all_sequences.columns.get_loc('Seq')]
        uniprot = self.all_sequences.iloc[idx, self.all_sequences.columns.get_loc('UniProt')]
        name = self.all_sequences.iloc[idx, self.all_sequences.columns.get_loc('Name')]
        labels=self.all_sequences.iloc[idx,self.all_sequences.columns.get_loc('Label')]
        sample={'idx':torch.tensor(idx).float(), 'seq':sequences,'label':torch.tensor(labels).int(),'UniProt':uniprot,'Name':name,'Loc':get_loc_from_name(name)} #multiple or single?
        if self.transform:
            sample=self.transform(sample)
        return sample

    def shuffle(self):
        print('dataset shuffled')
        return self.all_sequences.sample(frac=1,random_state=self.random_seed)

    def sort(self,ascending=True):
        s = self.all_sequences.Seq.str.len().sort_values(ascending=ascending).index
        df=self.all_sequences.reindex(s)
        self.all_sequences=df.reset_index(drop=True)

        return df

class ProteinWildSequence(Dataset):
    def __init__(self,file_path='/scratch/user/zshuying/ppi_mutation/ppi_seq_huri_humap.csv'):
  
        self.df_ppi=pd.read_csv(file_path)

    def __len__(self):
        return len(self.df_ppi)
    
    def __getitem__(self,idx):
        uniprot,sequence=self.df_ppi.iloc[idx,:]['UniProt'],self.df_ppi.iloc[idx,:]['seq']
        return{'seq':sequence,'UniProt':uniprot}




def cut_seq(seqDataset,low,medium,high,veryhigh,discard):
    curSeq=copy.deepcopy(seqDataset)
    curSeq.all_sequences=seqDataset.dataset.all_sequences.iloc[seqDataset.indices,:] # Subset object produced a bunch of random indices to call the __getitem__ from the original object. we have to match the indices
    shortSeq,mediumSeq,longSeq=copy.deepcopy(curSeq),copy.deepcopy(curSeq),copy.deepcopy(curSeq)

    s = curSeq.all_sequences.Seq.str.len()
    if high is None:
        high=np.inf
    curSeq.high=high
    curSeq.medium=medium
    curSeq.low=low
    cond1 = s >= low
    cond2 = s < medium
    cond3 = s >=medium
    cond4=s<high
    cond5=s>=high
    cond6=s<veryhigh
    cond7=s>veryhigh
    if discard:
        print('Discarded %s sequences which are longer than %s'%(len(curSeq.all_sequences[cond7]),veryhigh))
    shortSeq.indices = curSeq.all_sequences[cond1 & cond2].index
    print('short dataset cut to length between %s and %s' %(low,medium))
    print('sequences count = %s'%shortSeq.__len__())
    print('----------------------------------------------')

    mediumSeq. indices = curSeq.all_sequences[cond3 & cond4].index
    print('medium dataset cut to length between %s and %s' %(medium,high))
    print('sequences count = %s'%mediumSeq.__len__())
    print('----------------------------------------------')

    longSeq. indices = curSeq.all_sequences[cond5 & cond6].index
    print('long dataset cut to length between %s and %s' %(high,veryhigh))
    print('sequences count = %s'%longSeq.__len__())
    print('----------------------------------------------')
    if discard:
        return shortSeq,mediumSeq,longSeq
    if not discard:
        extremelongSeq=copy.deepcopy(curSeq)
        extremelongSeq. indices = curSeq.all_sequences[cond7].index
        print('extremelong dataset cut to length between %s and %s' %(veryhigh,np.inf))
        print('sequences count = %s'%extremelongSeq.__len__())
        print('----------------------------------------------')
        return shortSeq,mediumSeq,longSeq,extremelongSeq






class ProteinDataModule(pl.LightningDataModule):
    def __init__(self, low=0,medium=0,high=0,veryhigh=0,train_val_ratio=0.9,discard=True,crop_val=False,bs_short=4,bs_medium=2,bs_long=1,num_devices=1,num_nodes=1,delta=True,crop_len=False,which_dl=None,clinvar_csv=os.path.join(script_path,'merged_2019_1.csv'),mix_val=False,train_mix=False,random_seed=42,test=False,):
        super().__init__()
        self.dataset=ProteinSequence(clinvar_csv=clinvar_csv,delta=delta,random_seed=random_seed)
        self.crop_len=crop_len
        self.train_mix=train_mix
        self.which_dl=which_dl
        self.max_short=medium
        self.max_medium=high
        self.max_long=veryhigh
        self.mix_val=mix_val
        self.crop_val=crop_val
        self.seed=random_seed
        self.test=test
        self.gen_dataloader(train_val_ratio,low,medium,high,veryhigh,num_devices,num_nodes,bs_short,bs_medium,bs_long,train_mix)
        
    def gen_dataloader(self,train_val_ratio,low,medium,high,veryhigh,num_devices,num_nodes,bs_short,bs_medium,bs_long,train_mix):
        if self.test:
            print('Splitting train val with ratio = %s, did not seperate training set with lengths'%train_val_ratio)
            train_set,val_set= split_train_val(self.dataset,train_val_ratio,random_seed=self.seed)
                
            self.val_mix_dataloader = DataLoader(val_set, batch_size=bs_long,
                                            shuffle=False, num_workers=8,drop_last=True)

        else:
            if train_mix:
                print('Splitting train val with ratio = %s, did not seperate training set with lengths'%train_val_ratio)
                train_set,val_set= split_train_val(self.dataset,train_val_ratio,random_seed=self.seed)
                self.train_mix_dataloader = DataLoader(train_set, batch_size=bs_long,
                                                shuffle=True, num_workers=8,drop_last=True)
                self.val_mix_dataloader = DataLoader(val_set, batch_size=bs_long,
                                            shuffle=False, num_workers=8,drop_last=True)
                self.train_batch_num=len(train_set)//(num_devices*num_nodes*bs_short)
                self.val_batch_num=len(val_set)//(num_devices*num_nodes*bs_short)
            
            else:
                train_set,val_set= split_train_val(self.dataset,train_val_ratio,random_seed=self.seed)
                print('Splitting training set by length\n=======================')
                train_short_set,train_medium_set,train_long_set=cut_seq(train_set,low,medium,high,veryhigh,True)
                train_short_len,train_medium_len,train_long_len=len(train_short_set),len(train_medium_set),len(train_long_set),
                
                
                
                
                print('Splitting validation set by length\n=======================')
                val_short_set,val_medium_set,val_long_set=cut_seq(val_set,low,medium,high,veryhigh,True)
                val_mix_set,_,_=cut_seq(val_set,low,veryhigh,veryhigh+1,veryhigh+2,True)
                val_short_len,val_medium_len,val_long_len=\
                len(val_short_set),len(val_medium_set),len(val_long_set)

                
                if self.crop_val:val_mix_ds=2 
                else: val_mix_ds=1
                #make sure each machine gets the same num of batches otherwise it will hang
                self.ts, self.tm, self.tl, self.vs, self.vm, self.vl=\
                                train_short_len//(num_devices*num_nodes*bs_short),\
                                train_medium_len//(num_devices*num_nodes*bs_medium),\
                                train_long_len//(num_devices*num_nodes*bs_long),\
                                val_short_len//(num_devices*num_nodes*bs_short),\
                                val_medium_len//(num_devices*num_nodes*bs_medium),\
                                val_long_len//(num_devices*num_nodes*bs_long)

                if train_short_len:
                    self.train_short_dataloader = DataLoader(train_short_set, batch_size=bs_short,
                                                        shuffle=True, num_workers=20,drop_last=True)
                    
                    self.train_medium_dataloader = DataLoader(train_medium_set, batch_size=bs_medium,
                                                        shuffle=True, num_workers=20,drop_last=True)
                    self.train_long_dataloader = DataLoader(train_long_set, batch_size=bs_long,
                                                    shuffle=True, num_workers=20,drop_last=True)
                self.val_short_dataloader = DataLoader(val_short_set, batch_size=1,
                                                shuffle=False, num_workers=20,drop_last=True)
                self.val_medium_dataloader = DataLoader(val_medium_set, batch_size=1,
                                                shuffle=False, num_workers=20,drop_last=True)
                
                self.val_long_dataloader = DataLoader(val_long_set, batch_size=1,
                                                shuffle=False, num_workers=20,drop_last=True)
                self.val_mix_dataloader=DataLoader(val_mix_set, batch_size=val_mix_ds,
                                                shuffle=False, num_workers=20,drop_last=True)

    def train_dataloader(self):
        current_epoch=self.trainer.current_epoch
        if self.train_mix:
            self.trainer.limit_train_batches=self.train_batch_num-1
            self.which_dl='mix'
            return self.train_mix_dataloader
        if self.which_dl=='short':
            self.trainer.limit_train_batches=self.ts-1
            self.trainer.which_dl='short'
            print('short dataset')
            return self.train_short_dataloader
        elif self.which_dl=='medium':
            self.trainer.limit_train_batches=self.tm-1
            self.trainer.which_dl='medium'
            print('medium dataset')

            return self.train_medium_dataloader
        if self.which_dl=='long':
            self.trainer.limit_train_batches=self.tl-1
            self.trainer.which_dl='long'
            print('long dataset')
            return self.train_long_dataloader
        
        print('=======current epoch: %s =============='%current_epoch)
        if 0<=current_epoch%9<3:
            self.trainer.limit_train_batches=self.ts-1
            self.trainer.which_dl='short'
            return self.train_short_dataloader
        elif 3<=current_epoch%9<6:
            self.trainer.which_dl='medium'
            self.trainer.limit_train_batches=self.tm-1
            return self.train_medium_dataloader
        elif 6<=current_epoch%9<9:
            self.trainer.which_dl='long'
            self.trainer.limit_train_batches=self.tl-1
            return self.train_long_dataloader

    def val_dataloader(self):
        if self.mix_val:
            self.trainer.limit_val_batches=self.val_batch_num-1

            print('Evaluating validation with mixture of short,medium,long seqs')

            return self.val_mix_dataloader

        if self.which_dl=='short':
            self.trainer.limit_val_batches=self.ts-1
            self.trainer.which_dl='short'
            print('short dataset')
            return self.val_short_dataloader
        elif self.which_dl=='medium':
            self.trainer.limit_val_batches=self.tm-1
            self.trainer.which_dl='medium'
            print('medium dataset')
            return self.val_medium_dataloader
        if self.which_dl=='long':
            self.trainer.limit_val_batches=self.tl-1
            self.trainer.which_dl='long'
            print('long dataset')
            return self.val_long_dataloader
        


        current_epoch=self.trainer.current_epoch
        if 0<=current_epoch%9<3:
            self.trainer.limit_val_batches=self.vs-1
            # self.trainer.limit_val_batches=4
            return self.val_short_dataloader
        elif 3<=current_epoch%9<6:
            self.trainer.limit_val_batches=self.vm-1
            # self.trainer.limit_val_batches=5
            return self.val_medium_dataloader
        elif 6<=current_epoch%9<9:
            self.trainer.limit_val_batches=self.vl-1
            return self.val_long_dataloader

    def test_dataloader(self) :
        return self.val_mix_dataloader


class AllProteinVariantData(ProteinDataModule):
    """
    Subclass of ProteinSequence, without dividing train,val,test
    """

    def __init__(self, clinvar_csv=os.path.join(script_path, 'merged_2019_1.csv'),batch_size=20,num_workers=15):
        super().__init__(clinvar_csv=clinvar_csv)
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.dataset=ProteinSequence(clinvar_csv=clinvar_csv,delta=True)
    def val_dataloader(self):
        return DataLoader(self.dataset,batch_size=self.batch_size,shuffle=False, num_workers=self.num_workers,drop_last=True)
    def test_dataloader(self):
        return DataLoader(self.dataset,batch_size=self.batch_size,shuffle=False, num_workers=self.num_workers,drop_last=True)

class AllWildData(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.dataset=ProteinWildSequence()
    def train_dataloader(self) :
        return DataLoader(self.dataset,shuffle=False,batch_size=20,num_workers=15)
    def val_dataloader(self) :
        return DataLoader(self.dataset,shuffle=False,batch_size=20,num_workers=15)

class EsmMeanEmbeddings(Dataset):
    def __init__(self,if_initial_merge=False,dirpath=data_path):
        self.if_initial_merge=if_initial_merge
        self.dirpath=dirpath
        if self.if_initial_merge:self.initial_merge()
        self.read_file()


    def initial_merge(self):
        pattern = r'all.*predictions_[0-9]+.*'
        # Compile the regular expression
        regex = re.compile(pattern)
        # Loop over all files in the directory
        preds=[]
        for file in os.listdir(self.dirpath):
            if os.path.isfile(os.path.join(self.dirpath, file)):
                print(file)
                # Check if the file matches the pattern
                if regex.search(file):
                    pred=torch.load(os.path.join(self.dirpath, file))
                    preds.append(pred)
                    del pred
                    print('file %s has been loaded '%file)
        preds=torch.vstack(preds)
        print('predictions merged')
        torch.save(preds,os.path.join(data_path,'2019_all_esm_embeddings.pt'))
        print('predictions saved')
        pattern = 'all.*_labels_[0-9]+.*'
        # Compile the regular expression
        regex = re.compile(pattern)
        # Loop over all files in the directory
        preds=[]
        for file in os.listdir(self.dirpath):
            if os.path.isfile(os.path.join(self.dirpath, file)):
                # Check if the file matches the pattern
                if regex.search(file):
                    pred=torch.load(os.path.join(self.dirpath, file))
                    preds.append(pred)
                    del pred
                    print('file %s has been loaded '%file)

        preds=np.concatenate(preds,axis=0)
        print('labels merged')
        torch.save(preds,os.path.join(data_path,'2019_all_labels_for_embeddings.pt'))
        print('labels saved')
    def read_file(self):

        self.embeddings=torch.load(os.path.join(data_path,'2019_all_esm_embeddings.pt'))
        self.labels=torch.load(os.path.join(data_path,'2019_all_labels_for_embeddings.pt'))
        print(self.embeddings.shape)
        print(self.labels.shape)
        self.labels=(self.labels+1)/2

    def __len__(self):
        assert len(self.labels)==len(self.embeddings)
        return len(self.labels)
    def __getitem__(self, idx):
        return {'embedding':self.embeddings[idx,:],
                'label':self.labels[idx]}



def split_train_val(dataset,train_val_split=0.8,random_seed=52):
    train_set_size=int(len(dataset)*train_val_split)
    valid_set_size=len(dataset)-train_set_size
    seed=torch.Generator().manual_seed(random_seed)
    train_set,valid_set=data.random_split(dataset,[train_set_size,valid_set_size],generator=seed)
    print('Split dataset into train, val with the rate of %s'%train_val_split)
    print(train_set.dataset.all_sequences['Label'].value_counts())
    return train_set,valid_set