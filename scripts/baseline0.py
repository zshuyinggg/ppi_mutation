"""
This script is to develop models for single protein -- phenotype prediction
"""
import pandas as pd
from utils import *

def get_uniprot(uniprot_kb_list):
    return [item.split('#')[0] for item in uniprot_kb_list]

def gen_mutants(clinvar_file,out_file):
    df=pd.read_csv(clinvar_file,delimiter=';')
    uniprot_ls=get_uniprot(df['uniprot_kb']) 
    hgvs_ls=df['hgvs_p'].tolist()
    seq_file=gen_mutants_batch(uniprot_ls,hgvs_ls,1280,out_file)
    
gen_mutants('/home/grads/z/zshuying/Documents/shuying/ppi_mutation/data/clinvar/clinvar_pos.csv','/home/grads/z/zshuying/Documents/shuying/ppi_mutation/data/single_protein_seq/mutants_pos_bs1280')
gen_mutants('/home/grads/z/zshuying/Documents/shuying/ppi_mutation/data/clinvar/clinvar_neg.csv','/home/grads/z/zshuying/Documents/shuying/ppi_mutation/data/single_protein_seq/mutants_neg_bs1280')