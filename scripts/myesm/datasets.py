import copy

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

from scripts.utils import *
import pandas as pd
import dask.dataframe as ddf
import multiprocessing
from torchvision import transforms, utils

# num_partitions = multiprocessing.cpu_count()-4
num_partitions = 28





class ProteinSequence(Dataset):
    """
    Generate protein sequences according to provided clinvar_csv to gen_dir,
    label decides only positive samples(1), only negative samples(0), or both (None)
    """

    def __init__(self, clinvar_csv=os.path.join(script_path, 'merged_2019_1.csv'), gen_file_path=data_path + '/2019_1_sequences_terminated.csv', gen_file=False,
                 all_uniprot_id_file=os.path.join(data_path,'single_protein_seq/uniprotids_humap_huri.txt'),
                 test_mode=False,
                 transform=None,
                 random_seed=52,
                 delta=True):
        """
        labels: 2 wild. 0 negative. 1 possitive
        :param clinvar_csv:
        :param gen_file_path:
        :param gen_file:
        :param all_uniprot_id_file:
        :param test_mode:
        :param transform:
        :param train_val_ratio: ration for training set
        :param train_or_val: 'Train' or 'Val'
        :param random_seed:
        """
        self.random_seed=random_seed
        print('Reading file')
        self.clinvar = pd.read_csv(clinvar_csv)
        self.test_mode=test_mode
        self.gen_file_path = gen_file_path
        self.gen_file=gen_file
        self.transform=transform
        self.all_ppi_uniprot_ids = eval(open(all_uniprot_id_file).readline())
        self.clinvar = self.clinvar[
            [uniprot in self.all_ppi_uniprot_ids for uniprot in self.clinvar['UniProt'].tolist()]]
        if test_mode: 
            self.clinvar=self.clinvar.loc[:10,:]
        if gen_file:self.gen_sequence_file()
        else:self.read_sequence_file()
        if test_mode:self.all_sequences=self.all_sequences[:10]
        self.remove_na()
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        self.all_sequences=self.all_sequences.sample(frac=1,random_state=random_seed).reset_index(drop=True)
        #set this to class property to make sure train and val are split on the same indexes
        self.all_sequences.loc[self.all_sequences['Name']=='0','Label']=0 #wild type is considered the same as benign
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
    def cut_seq(self,low,high,discard):
        self.high=high
        self.low=low
        s = ProteinSequence.all_sequences.Seq.str.len()
        if high is None:
            high=np.inf
        if discard:
            print('Discarded %s sequences which are longer than %s'%(len(ProteinSequence.all_sequences[s>=self.discard_cutoff]),self.discard_cutoff))
        cond1 = s < self.high
        cond2 = s >= self.low
        self.all_sequences = self.ProteinSequence[cond1 & cond2]
        print('dataset cut to length between %s and %s' %(self.low,self.high))
        print('sequences count = %s'%self.__len__())
        print('----------------------------------------------')

    def set_class_seq(self):
        ProteinSequence.all_sequences=self.all_sequences

    def correct_labels(self):

        # self.all_sequences['Label']=(self.all_sequences['Label']+1)/2
        self.all_sequences.loc[self.all_sequences['Name']=='0','Label']=2
        print(self.all_sequences['Label'].describe())
        # self.all_sequences.to_csv(self.gen_file_path)

    def read_sequence_file(self):
        if os.path.isfile(self.gen_file_path):
            self.all_sequences = pd.read_csv(self.gen_file_path)
            self.all_sequences=self.all_sequences[self.all_sequences['Seq'].apply(lambda x: not('Error' in x))]
            self.all_sequences.reset_index(drop=True, inplace=True)
        else: self.gen_sequence_file()
    def gen_sequence_file(self) -> object:
        if self.test_mode:print('-----Test mode on-----------')
        print('Initiating datasets....\n')
        print('Generating mutant sequences...\n')
        df_sequence_mutant = self.clinvar.loc[:, ['#AlleleID', 'label', 'UniProt', 'Name']]  # TODO review status
        df_sequence_mutant=df_sequence_mutant[df_sequence_mutant['UniProt'].isin(self.all_ppi_uniprot_ids)]
        print('There are %s rows'%len(df_sequence_mutant))
        # df_sequence_mutant['Seq'] = [gen_mutant_one_row(uniprot_id, name) for uniprot_id, name in \
        #                              zip(df_sequence_mutant['UniProt'], df_sequence_mutant['Name'])]
        df_dask = ddf.from_pandas(df_sequence_mutant, npartitions=num_partitions)
        print('lazy partitions set')
        df_dask['Seq'] = df_dask.map_partitions(gen_mutant_from_df, meta=('str'))
        df_sequence_mutant=df_dask.compute(scheduler='multiprocessing')
        len_wild = len(self.all_ppi_uniprot_ids)
        df_sequence_mutant.to_csv(self.gen_file_path)
        df_sequence_mutant=pd.read_csv(self.gen_file_path)
        # del df_dask
        print('Generating wild sequences...\n')
        self.all_ppi_uniprot_ids=list(self.all_ppi_uniprot_ids)
        if self.test_mode:
            self.all_ppi_uniprot_ids=self.all_ppi_uniprot_ids[:100]
            len_wild = 100

        df_sequence_wild = pd.DataFrame(0, index=np.arange(len_wild),
                                        columns=['#AlleleID', 'Label', 'UniProt', 'Name', 'Seq'])
        df_sequence_wild['UniProt'] = list(self.all_ppi_uniprot_ids)
        df_dask = ddf.from_pandas(df_sequence_wild, npartitions=num_partitions)
        df_dask['Seq'] = df_dask.map_partitions(get_sequence_from_df, meta=('str'))
        df_sequence_wild=df_dask.compute(scheduler='multiprocessing')

        # df_sequence_wild['Seq'] = [get_sequence_from_uniprot_id(id) for id in df_sequence_wild['UniProt']]
        df_sequence_wild['Label'] = [-1] * len_wild
        df_sequences = pd.concat([df_sequence_wild, df_sequence_mutant])
        df_sequences.to_csv(self.gen_file_path)
        self.all_sequences = df_sequences # TODO
        return df_sequences


    def __len__(self):
        return len(self.all_sequences)

    def __getitem__(self, idx, uniprot=None, label=None):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sequences = self.all_sequences.iloc[idx, self.all_sequences.columns.get_loc('Seq')]
        uniprot = self.all_sequences.iloc[idx, self.all_sequences.columns.get_loc('UniProt')]
        name = self.all_sequences.iloc[idx, self.all_sequences.columns.get_loc('Name')]
        labels=self.all_sequences.iloc[idx,self.all_sequences.columns.get_loc('Label')]
        sample={'idx':torch.tensor(idx).float(), 'seq':sequences,'label':torch.tensor(labels).int(),'UniProt':uniprot,'Name':name} #multiple or single?
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



class ProteinSequencePair(ProteinSequence):
        def __init__(self):
            """
            :param clinvar_csv:
            :param gen_file_path:
            :param gen_file:
            :param all_uniprot_id_file:
            :param test_mode:
            :param transform:
            :param train_val_ratio: ration for training set
            :param train_or_val: 'Train' or 'Val'
            :param random_seed:
            """
            super().__init__(clinvar_csv=os.path.join(script_path, 'merged_2019_1.csv'), gen_file_path=data_path + '/2019_1_sequences_terminated.csv', gen_file=False,
                 all_uniprot_id_file=os.path.join(data_path,'single_protein_seq/uniprotids_humap_huri.txt'),
                 test_mode=False,
                 transform=None,
                 random_seed=52)
        def __getitem__(self, idx=None, uniprot=None, label=None):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            sequences = self.all_sequences.iloc[idx, self.all_sequences.columns.get_loc('Seq')]
            labels=self.all_sequences.iloc[idx,self.all_sequences.columns.get_loc('Label')]
            sample={'idx':torch.tensor(idx).float(), 'seq':sequences,'label':torch.tensor(labels).int(),'Name':self.all_sequences.iloc[idx,self.all_sequences.columns.get_loc('Name')],'UniProt':self.all_sequences.iloc[idx,self.all_sequences.columns.get_loc('UniProt')]} #multiple or single?
            return sample



        def get_idx_from_uniprot(self,uniprot):
            idx = self.all_sequences[self.all_sequences['UniProt']==uniprot].index
            return idx



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
    def __init__(self, low,medium,high,veryhigh,train_val_ratio=0.9,discard=True,bs_short=4,bs_medium=2,bs_long=1,num_devices=1,num_nodes=1,delta=True,random_crop_len=False,which_dl=None):
        super().__init__()
        self.dataset=ProteinSequence(delta=delta)
        self.random_crop_len=random_crop_len
        self.which_dl=which_dl
        self.gen_dataloader(train_val_ratio,low,medium,high,veryhigh,num_devices,num_nodes,bs_short,bs_medium,bs_long)
        
    def gen_dataloader(self,train_val_ratio,low,medium,high,veryhigh,num_devices,num_nodes,bs_short,bs_medium,bs_long):
        train_set,val_set= split_train_val(self.dataset,train_val_ratio)
        print('Splitting training set by length\n=======================')
        train_short_set,train_medium_set,train_long_set=cut_seq(train_set,low,medium,high,veryhigh,True)
        print('Splitting validation set by length\n=======================')
        val_short_set,val_medium_set,val_long_set=cut_seq(val_set,low,medium,high,veryhigh,True)

        train_short_len,train_medium_len,train_long_len,\
        val_short_len,val_medium_len,val_long_len=\
        len(train_short_set),len(train_medium_set),len(train_long_set),\
        len(val_short_set),len(val_medium_set),len(val_long_set)

        #make sure each machine gets the same num of batches otherwise it will hang
        self.ts, self.tm, self.tl, self.vs, self.vm, self.vl=\
                        train_short_len//(num_devices*num_nodes*bs_short),\
                        train_medium_len//(num_devices*num_nodes*bs_medium),\
                        train_long_len//(num_devices*num_nodes*bs_long),\
                        val_short_len//(num_devices*num_nodes*bs_short),\
                        val_medium_len//(num_devices*num_nodes*bs_medium),\
                        val_long_len//(num_devices*num_nodes*bs_long)


        self.train_short_dataloader = DataLoader(train_short_set, batch_size=bs_short,
                                            shuffle=True, num_workers=20,drop_last=True)
        self.val_short_dataloader = DataLoader(val_short_set, batch_size=bs_short,
                                          shuffle=False, num_workers=20,drop_last=True)
        self.train_medium_dataloader = DataLoader(train_medium_set, batch_size=bs_medium,
                                             shuffle=True, num_workers=20,drop_last=True)
        self.val_medium_dataloader = DataLoader(val_medium_set, batch_size=bs_medium,
                                           shuffle=False, num_workers=20,drop_last=True)
        self.train_long_dataloader = DataLoader(train_long_set, batch_size=bs_long,
                                           shuffle=True, num_workers=20,drop_last=True)
        self.val_long_dataloader = DataLoader(val_long_set, batch_size=bs_long,
                                         shuffle=False, num_workers=20,drop_last=True)


    def train_dataloader(self):
        current_epoch=self.trainer.current_epoch

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


class toHF(object):
    """add space in between each amino acid in order to put it in tokenizer of huggingface"""
    def __call__(self,sample):
        sequences=sample['seq']
        sequences=" ".join(list(sequences))
        sample['seq']=sequences 
        return sample


class ToTensor(object):
    """convert pandas object to Tensors"""

    def __call__(self,sample):
        idx,sequences,labels=sample['idx'],sample['seq'],sample['label']
        return {'idx':torch.tensor(idx),
        'seq':torch.tensor(sequences),
        'label':torch.tensor(labels)}


class RandomCrop(object):
    def __init__(self,crop_size):
        assert isinstance(crop_size,int)
        self.crop_size=crop_size

    def __call__(self,sample):
        sequence=sample['seq']
        l=len(sequence)
        if self.crop_size<l:
            start=np.random.randint(0,l-self.crop_size)
        else:
            start=0
        new_seq=sequence[start:]
        return {'idx':sample['idx'], 'seq':new_seq,'label':sample['label']}



def split_train_val(dataset,train_val_split=0.8,random_seed=52):
    train_set_size=int(len(dataset)*train_val_split)
    valid_set_size=len(dataset)-train_set_size
    seed=torch.Generator().manual_seed(random_seed)
    train_set,valid_set=data.random_split(dataset,[train_set_size,valid_set_size],generator=seed)
    print('Split dataset into train, val with the rate of %s'%train_val_split)
    return train_set,valid_set