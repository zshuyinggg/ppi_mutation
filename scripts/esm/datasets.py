import torch
from torch.utils.data import Dataset
from os import path
import sys

sys.path.append((path.dirname(path.dirname(path.dirname( path.abspath(__file__))))))
print(path.dirname(path.dirname(path.dirname( path.abspath(__file__)))))
from scripts.utils import *

import pandas as pd


# setting path

class ProteinSequence(Dataset):
    """
    Generate protein sequences according to provided clinvar_csv to gen_dir, 
    label decides only positive samples(1), only negative samples(0), or both (None)
    """
    def __init__(self, clinvar_csv, gen_dir,label=None):
        self.clinvar=pd.read_csv(clinvar_csv)
        self.gen_dir=gen_dir
        self.label=label
        self.positive_sequence_file, self.negative_sequence_file=self.gen_sequence_file (label)

    def gen_sequence_file(label):
        pass


        
    def __len__(self):
        return len(self.clinvar.index)
    
    def __getitem__(self, idx, uniprot=None, label=None):
        if torch.is_tensor(idx):
            idx=idx.tolist()


def check_if_in_ppi(uniprotID):
    f_path=os.path.join(top_path,'data/single_protein_seq/uniprotids_humap_huri.txt')
    f=eval(open(f_path,'r').readline())
    if uniprotID in f: return True
    else: return False

# check_if_in_ppi('sss')