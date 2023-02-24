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
    def __init__(self, clinvar_csv, gen_dir, tokenizer, all_uniprot_id_file='data/single_protein_seq/uniprotids_humap_huri.txt', label=None):
        check_sep_clinvar(clinvar_csv)
        self.clinvar=pd.read_csv(clinvar_csv)
        self.gen_dir=gen_dir
        self.tokenizer=tokenizer
        self.label=label
        self.all_uniprot_id_file=all_uniprot_id_file
        self.gen_sequence_file(label)

    def gen_sequence_file(self,label):
        self.clinvar['label']=if_positive_or_negative(self.clinvar['clinical_sig'])
        self.wild_uniprotIDs=eval(open(os.path.join(top_path,self.all_uniprot_id_file),'r').readline())
        self.positive_data=self.clinvar[self.clinvar['label']==1]
        self.positive_uniprotIDs=get_uniprot(self.positive_data['uniprot_kb'])
        self.negative_data=self.clinvar[self.clinvar['label']==-1]
        self.negative_uniprotIDs=get_uniprot(self.negative_data['uniprot_kb'])
        self.positive_sequences=gen_mutants(self.positive_data,self.gen_dir+'positive',bs=None)
        self.negative_sequences=gen_mutants(self.negative_data,self.gen_dir+'negative',bs=None)
        self.wild_sequences=gen_sequences_oneFile(list(self.wild_uniprotIDs),self.gen_dir+'wild')
        # return self.positive_sequences,self.negative_sequences,self.wild_sequences
    
    def __len__(self):
        return len(self.clinvar.index)
    
    def __getitem__(self, idx, uniprot=None, label=None):
        if torch.is_tensor(idx):
            idx=idx.tolist()
        if not uniprot:
            if label==1: #all positive
                sequences_uniprotIDs=self.positive_uniprotIDs[idx]
                sequences=[' '.join(self.positive_sequences[x]) for x in sequences_uniprotIDs]
            elif label==-1:#all negative
                sequences_uniprotIDs=self.negative_uniprotIDs[idx]
                sequences=[' '.join(self.negative_sequences[x]) for x in sequences_uniprotIDs]                
            elif label==0:#all wildtype
                sequences_uniprotIDs=self.wild_uniprotIDs[idx]
                sequences=[' '.join(self.wild_sequences[x]) for x in sequences_uniprotIDs]
        

def check_if_in_ppi(uniprotID):
    f_path=os.path.join(top_path,'data/single_protein_seq/uniprotids_humap_huri.txt')
    f=eval(open(f_path,'r').readline())
    if uniprotID in f: return True
    else: return False



# check_if_in_ppi('sss')