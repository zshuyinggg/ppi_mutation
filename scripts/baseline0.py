"""
This script is to develop models for single protein -- phenotype prediction
"""
import pandas as pd
from utils import *

def gen_mutants(clinvar_file,out_file):
    df=pd.read_csv(clinvar_file,delimiter=';')
    uniprot_ls=df['uniprot_kb']
    hgvs_ls=df['hgvs_p']
    seq_file=gen_mutants_batch(uniprot_ls,hgvs_ls,1280,out_file)
    