"""
This script is to develop models for single protein -- phenotype prediction
"""
from utils import *

    
gen_mutants('/home/grads/z/zshuying/Documents/shuying/ppi_mutation/data/clinvar/clinvar_pos.csv','/home/grads/z/zshuying/Documents/shuying/ppi_mutation/data/single_protein_seq/mutants_pos_bs1280')
gen_mutants('/home/grads/z/zshuying/Documents/shuying/ppi_mutation/data/clinvar/clinvar_neg.csv','/home/grads/z/zshuying/Documents/shuying/ppi_mutation/data/single_protein_seq/mutants_neg_bs1280')