from tqdm import tqdm
import multiprocessing
import sys
from utils import *
import pandas as pd
import os
test=False

pj=os.path.join
def boolean_review_status(pd_series,level):
    """
    ['criteria provided, single submitter',
    'criteria provided, multiple submitters, no conflicts',
    'reviewed by expert panel', 'practice guideline']
    """
    if level==1:
            l=['criteria provided, single submitter','criteria provided, multiple submitters, no conflicts',
    'reviewed by expert panel', 'practice guideline']
    elif level==2:
            l=['criteria provided, multiple submitters, no conflicts',
    'reviewed by expert panel', 'practice guideline']
    elif level==3:
            l=['reviewed by expert panel', 'practice guideline']
    else:
            l=['practice guideline']
    pd_bool=pd_series.apply(lambda x:x in l)
    print('Included items with review status as %s, length=%s'%(l,sum(pd_bool)) )
    return pd_bool



# initialize the data files
num_processes=17
print('%s processes'%num_processes)
merged_clinvar_csv,save_name, review_status=pj(top_path,'scripts/merged_2023_2_26.csv'),\
    pj(top_path,'data/seqs_2023_2_26.csv'), 2
ref_uniprot_file='/home/grads/z/zshuying/Documents/shuying/ppi_mutation/data/single_protein_seq/uniprotids_humap_huri.txt'
f=pd.read_csv(merged_clinvar_csv)
if test: f=f.iloc[:100,:]
all_ppi_uniprot_ids = eval(open(ref_uniprot_file).readline()) # all uniprot ids that involve in PPI of humap and huri
f = f[[uniprot in all_ppi_uniprot_ids for uniprot in f['UniProt'].tolist()]]
print(f['ReviewStatus'].value_counts())
f=f[boolean_review_status(f['ReviewStatus'],review_status)]
f_sequence_mutant = f.loc[:, ['#AlleleID', 'label', 'UniProt', 'Name']]
f_sequence_mutant=f_sequence_mutant[f_sequence_mutant['UniProt'].isin(all_ppi_uniprot_ids)].reset_index(drop=True)
print('There are %s rows'%len(f_sequence_mutant))
all_ppi_uniprot_ids=list(all_ppi_uniprot_ids)
len_wild = len(all_ppi_uniprot_ids)
f_sequence_wild = pd.DataFrame(0, index=np.arange(len_wild),
                                columns=['#AlleleID', 'Label', 'UniProt', 'Name', 'Seq'])
f_sequence_wild['UniProt'] = list(all_ppi_uniprot_ids)

def mp_func_mutant(df):
    mutant_df=Mutant_df(df)
    print('%s started'%os.getpid())
    df=mutant_df.gen_mutant_from_df()
    return df
    # df=mutant_df.merge_dfs()
def mp_func_wild(df):
    wild_df=Wild_df(df)
    print('%s started'%os.getpid())
    df=wild_df.get_sequence_from_df()
    return df
    # df=wild_df.merge_dfs()

chunk_size_mutant = int(f_sequence_mutant.shape[0]/num_processes)
chunks_mutant = [f_sequence_mutant.loc[f_sequence_mutant.index[i:i + chunk_size_mutant]] for i in range(0, f_sequence_mutant.shape[0], chunk_size_mutant)]
pool = multiprocessing.Pool(processes=num_processes)
result =pool.map(mp_func_mutant, chunks_mutant)
f_final_mutant=pd.concat(result,sort=False)
f_final_mutant.to_csv('mutant_seq_2023_2.csv')

# chunk_size_wild = int(f_sequence_wild.shape[0]/num_processes)
# chunks_wild = [f_sequence_wild.loc[f_sequence_wild.index[i:i + chunk_size_wild]] for i in range(0, f_sequence_wild.shape[0], chunk_size_wild)]
# pool = multiprocessing.Pool(processes=num_processes)
# result =pool.map(mp_func_wild, chunks_wild)
# f_final_wild=pd.concat(result,sort=False)
# f_final_wild.to_csv('wild_seq_2023_2.csv')


# f_final=pd.concat([f_final_mutant,f_final_wild],sort=False)
# f_final.to_csv('final_2023.csv')