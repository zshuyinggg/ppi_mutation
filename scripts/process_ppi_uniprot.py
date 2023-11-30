from tqdm import tqdm
import multiprocessing
import sys
import pandas as pd
from utils_clinvar import *
import os
test=False

pj=os.path.join
# initialize the data files
num_processes=64




def mp_func_ppi(df):
    ppi_df=MultiProcessUniProt(df=df)
    print('%s started'%os.getpid())
    try:df=ppi_df.process_ppi_df(save_name='ppi_seq')
    except KeyError: print(df)
    return df

if __name__ == '__main__':
    all_ppi_uniprot_ids = list(eval(open('/scratch/user/zshuying/ppi_mutation/data/single_protein_seq/uniprotids_humap_huri.txt').readline()))
    df_ppi=pd.DataFrame(columns=['UniProt','Seq'],dtype=object)
    df_ppi['UniProt']=all_ppi_uniprot_ids

    chunk_size_ppi = int(df_ppi.shape[0]/num_processes)
    # chunk_size_ppi = int(preprocess_variant_df.shape[0]/num_processes)
    chunks_ppi = [df_ppi.loc[df_ppi.index[i:i + chunk_size_ppi]] for i in range(0, df_ppi.shape[0], chunk_size_ppi)]
    pool = multiprocessing.Pool(processes=num_processes)
    result =pool.map(mp_func_ppi, chunks_ppi)
    f_final_ppi=pd.concat(result,sort=False)
    f_final_ppi.to_csv('ppi_seq_huri_humap.csv')

# chunk_size_wild = int(f_sequence_wild.shape[0]/num_processes)
# chunks_wild = [f_sequence_wild.loc[f_sequence_wild.index[i:i + chunk_size_wild]] for i in range(0, f_sequence_wild.shape[0], chunk_size_wild)]
# pool = multiprocessing.Pool(processes=num_processes)
# result =pool.map(mp_func_wild, chunks_wild)
# f_final_wild=pd.concat(result,sort=False)
# f_final_wild.to_csv('wild_seq_2023_2.csv')


# f_final=pd.concat([f_final_ppi,f_final_wild],sort=False)
# f_final.to_csv('final_2023.csv')