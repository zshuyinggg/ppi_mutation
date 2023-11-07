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




def mp_func_mutant(df):
    mutant_df=MultiProcessClinvar(df=df)
    print('%s started'%os.getpid())
    df=mutant_df.process_variants_df(save_name='mutant_seq_2022_2')
    return df

if __name__ == '__main__':
    preprocess_variant_sum_file=MultiProcessClinvar(variant_summary_file=pj(data_path,'clinvar/variant_summary_2020-06.txt'),\
                                                df=None,review_status=1)
    # preprocess_variant_df=pd.read_csv('/home/grads/z/zshuying/Documents/shuying/ppi_mutation/data/clinvar/mutant_seq_2022_2.csv')
    chunk_size_mutant = int(preprocess_variant_sum_file.df.shape[0]/num_processes)
    # chunk_size_mutant = int(preprocess_variant_df.shape[0]/num_processes)
    chunks_mutant = [preprocess_variant_sum_file.df.loc[preprocess_variant_sum_file.df.index[i:i + chunk_size_mutant]] for i in range(0, preprocess_variant_sum_file.df.shape[0], chunk_size_mutant)]
    pool = multiprocessing.Pool(processes=num_processes)
    result =pool.map(mp_func_mutant, chunks_mutant)
    f_final_mutant=pd.concat(result,sort=False)
    f_final_mutant.to_csv('mutant_seq_2020_6.csv')

# chunk_size_wild = int(f_sequence_wild.shape[0]/num_processes)
# chunks_wild = [f_sequence_wild.loc[f_sequence_wild.index[i:i + chunk_size_wild]] for i in range(0, f_sequence_wild.shape[0], chunk_size_wild)]
# pool = multiprocessing.Pool(processes=num_processes)
# result =pool.map(mp_func_wild, chunks_wild)
# f_final_wild=pd.concat(result,sort=False)
# f_final_wild.to_csv('wild_seq_2023_2.csv')


# f_final=pd.concat([f_final_mutant,f_final_wild],sort=False)
# f_final.to_csv('final_2023.csv')