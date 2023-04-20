import torch
from torch.utils.data import Dataset

global top_path  # the path of the top_level directory
global script_path, data_path, logging_path
import os, sys
from torch.utils.data import DataLoader
import esm
from dask.diagnostics import ProgressBar


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

    def __init__(self, clinvar_csv, gen_file_path, gen_file=True,
                 all_uniprot_id_file='data/single_protein_seq/uniprotids_humap_huri.txt',
                 test_mode=False,
                 transform=None):
        print('Reading file')
        self.clinvar = pd.read_csv(clinvar_csv)
        self.test_mode=test_mode
        self.gen_file_path = gen_file_path
        self.gen_file=gen_file
        self.all_ppi_uniprot_ids = eval(open(all_uniprot_id_file).readline())
        self.clinvar = self.clinvar[
            [uniprot in self.all_ppi_uniprot_ids for uniprot in self.clinvar['UniProt'].tolist()]]
        # check if sequence file already exist:
        if test_mode:
            self.clinvar=self.clinvar.loc[:100,:]
        if gen_file:
            self.gen_sequence_file()
        else:
            self.read_sequence_file()
        self.transform=transform
    def read_sequence_file(self):
        if os.path.isfile(self.gen_file_path):
            self.all_sequences = pd.read_csv(self.gen_file_path)
            self.all_sequences=self.all_sequences[self.all_sequences['Seq'].apply(lambda x: not('Error' in x))]
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
        #TODO exclude those who is not in ppi
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
        labels=self.all_sequences.iloc[idx,self.all_sequences.columns.get_loc('Label')]
        sample={'idx':idx, 'seq':sequences,'label':labels} #multiple or single?
        if self.transform:
            sample=self.transform(sample)
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



