import torch
from torch.utils.data import Dataset

global top_path  # the path of the top_level directory
global script_path, data_path, logging_path
import os, sys
from torch.utils.data import DataLoader
import esm


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

# num_partitions = multiprocessing.cpu_count()-4
num_partitions = 19

class ProteinSequence(Dataset):
    """
    Generate protein sequences according to provided clinvar_csv to gen_dir,
    label decides only positive samples(1), only negative samples(0), or both (None)
    """

    def __init__(self, clinvar_csv, gen_file_path, gen_file=True,
                 all_uniprot_id_file='data/single_protein_seq/uniprotids_humap_huri.txt',
                 test_mode=False):
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
            self.clinvar=self.clinvar.loc[:1000,:]
        if gen_file:
            self.gen_sequence_file()
        else:
            if os.path.isfile(gen_file_path):
                self.all_sequences = pd.read_csv(gen_file_path)
                self.all_sequences=self.all_sequences[self.all_sequences['Seq'].notna()]
            else:
                self.gen_sequence_file()

    def gen_sequence_file(self) -> object:
        if self.test_mode:print('-----Test mode on-----------')
        print('Initiating datasets....\n')
        print('Generating mutant sequences...\n')
        df_sequence_mutant = self.clinvar.loc[:, ['#AlleleID', 'label', 'UniProt', 'Name']]  # TODO review status
        # df_sequence_mutant['Seq'] = [gen_mutant_one_row(uniprot_id, name) for uniprot_id, name in \
        #                              zip(df_sequence_mutant['UniProt'], df_sequence_mutant['Name'])]
        df_dask = ddf.from_pandas(df_sequence_mutant, npartitions=num_partitions)
        df_dask['Seq'] = df_dask.map_partitions(gen_mutant_from_df, meta=('str')).compute(scheduler='multiprocessing')
        len_wild = len(self.all_ppi_uniprot_ids)
        df_sequence_mutant.to_csv(self.gen_file_path)
        df_sequence_mutant=pd.read_csv(self.gen_file_path)
        #TODO exclude those who is not in ppi

        print('Generating wild sequences...\n')
        self.all_ppi_uniprot_ids=list(self.all_ppi_uniprot_ids)
        if self.test_mode:
            self.all_ppi_uniprot_ids=self.all_ppi_uniprot_ids[:100]
            len_wild = 100

        df_sequence_wild = pd.DataFrame(0, index=np.arange(len_wild),
                                        columns=['#AlleleID', 'Label', 'UniProt', 'Name', 'Seq'])
        df_sequence_wild['UniProt'] = list(self.all_ppi_uniprot_ids)
        df_sequence_wild['Seq'] = [get_sequence_from_uniprot_id(id) for id in df_sequence_wild['UniProt']]
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
        sequences = self.all_sequences.loc[idx, 'Seq']  # TODO check index
        # labels=self.all_sequences.loc[idx,'Label']
        return idx, sequences.replace('*','')


# %%

