import sys
import numpy as np
from Bio.PDB.DSSP import dssp_dict_from_pdb_file

from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist #SZ add
import multiprocessing
matrix = matlist.blosum62

import re
import string
import random

import networkx as nx # for graph similarity


def sequence_indentity(seq_1, seq_2, version = 'BLAST'):
    '''Calculate the identity between two sequences
    :param seq_1, seq_2: protein sequences
    :type seq_1, seq_2: str
    :param version: squence identity version
    :type version: str, optional
    :return: sequence identity
    :rtype: float
    '''
    l_x = len(seq_1)
    l_y = len(seq_2)
    X = seq_1.upper()
    Y = seq_2.upper()

    if version == 'BLAST':
        alignments = pairwise2.align.globaldd(X,Y, matrix,-11,-1,-11,-1)   # Consistent with Blast P grobal alignment
        max_iden = 0
        for i in alignments:
            same = 0
            for j in range(i[-1]):
                if i[0][j] == i[1][j] and i[0][j] != '-':
                    same += 1
            iden = float(same)/float(i[-1])
            if iden > max_iden:
                max_iden = iden
        identity = max_iden
    elif version == 'Gap_exclude':
        l = min(l_x,l_y)
        alignments = pairwise2.align.globaldd(X,Y, matrix,-11,-1,-11,-1)   # Consistent with Blast P grobal alignment
        max_same = 0
        for i in alignments:
            same = 0
            for j in range(i[-1]):
                if i[0][j] == i[1][j] and i[0][j] != '-':
                    same += 1
            if same > max_same:
                max_same = same
        identity = float(max_same)/float(l)
    else:
        print('Error! No sequence identity version named %s!'%version)
    return identity

num_processes=48



import os
def mp_func_ppi(df):
    print('%s started, overall shape = %s'%(os.getpid(),str(df.shape)),flush=True)
    for i in range(df.shape[0]):
        print('pid: %s, working on %s'%(os.getpid(),i),flush=True)
        for j in range(df.shape[1]):
            if j%20==0:print('pid: %s,  %.2f done'%(os.getpid(),j/df.shape[1]),flush=True)
            df.iloc[i,j]=sequence_indentity(uniprot_seq_dict[df.index[i]],uniprot_seq_dict[df.columns[j]])
    return df

if __name__ == '__main__':

    import pandas as pd
    uniprot_seq=pd.read_csv('/scratch/user/zshuying/ppi_mutation/ppi_seq_huri_humap.csv')
    uniprot_seq_dict=dict(map(lambda i,j : (i,j) , uniprot_seq['UniProt'].tolist(),uniprot_seq['seq'].tolist()))
    test_set=pd.read_csv('/scratch/user/zshuying/ppi_mutation/data/clinvar/mutant_seq_2019_test_no_error.csv'
    )
    with open('/scratch/user/zshuying/ppi_mutation/data/baseline1/processed/2019_test_name_list_1050.txt','r') as f:
        test_list=eval(f.readline())
    with open('/scratch/user/zshuying/ppi_mutation/data/baseline1/processed/2019_train_name_list_1050.txt','r') as f:
        train_list=eval(f.readline())
    with open('/scratch/user/zshuying/ppi_mutation/data/baseline1/processed/2019_val_name_list_1050.txt','r') as f:
        val_list=eval(f.readline())
    train_val=pd.read_csv('/scratch/user/zshuying/ppi_mutation/data/clinvar/mutant_seq_2019_1_no_error.csv')

    test_actual_df=test_set[test_set['Name'].apply(lambda x:x in test_list)]
    test_uniprot=set(test_actual_df.UniProt.unique().tolist())
    train_actual_df=train_val[train_val['Name'].apply(lambda x:x in train_list)]
    train_uniprot=set(train_actual_df.UniProt.unique().tolist())
    val_actual_df=train_val[train_val['Name'].apply(lambda x:x in val_list)]
    val_uniprot=set(val_actual_df.UniProt.unique().tolist())
    unseen_test_uniprot=test_uniprot-train_uniprot-val_uniprot
    seq_sim_train_test=pd.DataFrame(index=list(train_uniprot),columns=list(unseen_test_uniprot))
    seq_sim_train_val=pd.DataFrame(index=list(train_uniprot),columns=list(val_uniprot))
    #train_vel
    chunk_size_ppi = int(seq_sim_train_val.shape[0]/num_processes)
    chunks_ppi = [seq_sim_train_val.iloc[i:i + chunk_size_ppi,:] for i in range(0, seq_sim_train_val.shape[0], chunk_size_ppi)]
    pool = multiprocessing.Pool(processes=num_processes)
    result =pool.map(mp_func_ppi, chunks_ppi)
    f_final_ppi=pd.concat(result,sort=False)
    f_final_ppi.to_csv('seq_sim_train_val.csv')

    # train_test
    chunk_size_ppi = int(seq_sim_train_test.shape[0]/num_processes)
    chunks_ppi = [seq_sim_train_test.iloc[i:i + chunk_size_ppi,:] for i in range(0, seq_sim_train_test.shape[0], chunk_size_ppi)]
    pool = multiprocessing.Pool(processes=num_processes)
    result =pool.map(mp_func_ppi, chunks_ppi)
    f_final_ppi=pd.concat(result,sort=False)
    f_final_ppi.to_csv('seq_sim_train_test.csv')
    print('train_test done')

