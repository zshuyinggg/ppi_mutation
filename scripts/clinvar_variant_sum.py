from tqdm import tqdm
import multiprocessing
from os import path
import sys
from utils import *

import pandas as pd
f=pd.read_csv(os.path.join(data_path,'clinvar','variant_summary_2023_02_26.txt'),sep='\t')
# f_simple=pd.read_csv('/home/grads/z/zshuying/Documents/shuying/ppi_mutation/scripts/2022-02-variant-sum_remain.csv',index_col='#AlleleID')
def if_positive_or_negative(string_list):
    label=[]
    for string in string_list:
        if string in ['Pathogenic', 
                    'Pathogenic/Likely pathogenic', 
                    'probable-pathogenic',
                    'Likely pathogenic', 
                    'pathologic', 
                    'pathogenic',
                    'likely pathogenic',
                    'Pathogenic/Likely pathogenic/Established risk allele',
                    'likely pathogenic - adrenal pheochromocytoma',
                    'Pathogenic/Pathogenic, low penetrance',
                    'Pathogenic, low penetrance']:
            label.append(1)
        elif string in ['Benign',
                    'Likely benign',
                    'Likely Benign',
                    'Benign/Likely benign',
                    'non-pathogenic', 
                    'benign', 'probable-non-pathogenic', 'Likely Benign', 'probably not pathogenic',
        ]:
            label.append(-1)

        else: label.append(0)
    return label
def extract_uniprot(string):
    try: m=re.match(r'.*UniProtKB:([0-9A-Z].*)#.*',string).group(1)
    except AttributeError: 
        print(string)
        m='0'
    return m


import requests
import json
import re
def find_fasta_refseq(refseqID):
    url='https://rest.uniprot.org/uniprotkb/search?query=%s'%refseqID
    cnt=json.loads(requests.get(url).text)
    try:
        assert re.search('uniprot',cnt.get("results")[0].get("entryType"),re.IGNORECASE)
        return cnt["results"][0]['primaryAccession']
    except:
        print('Error found in '+url)
        return 'Error found in '+url

def get_uniprot_from_name(name):
    try:
        refseq=re.match(r'\S*?(?=[\(:])',name).group(0)
        uniprot=find_fasta_refseq(refseq)

    except:
        print('Error Finding Refseq')
        uniprot=find_fasta_refseq(refseq)
    return uniprot 



f['label']=if_positive_or_negative(f['ClinicalSignificance'])

keep_cols=['#AlleleID', 'Type', 'Name','label', 'ClinicalSignificance', 'ClinSigSimple', 'ReviewStatus', 'OtherIDs', 'LastEvaluated', 'RS# (dbSNP)',
       'nsv/esv (dbVar)', 'RCVaccession']
keep_conditions=(f['Type']=='single nucleotide variant') &\
              (['no assertion' not in item for item in f['ReviewStatus'].tolist()]) &\
               (['p' in item for item in f['Name'].tolist()]) &\
               (f['label']!=0)
f_simple=f[keep_conditions][keep_cols]
f_simple=f_simple.drop_duplicates(subset=['#AlleleID'])

f_simple.head()


num_processes = multiprocessing.cpu_count()
print('%s cpus'%num_processes)
chunk_size = int(f_simple.shape[0]/num_processes)
# global pbar
# pbar = tqdm(total=len(f_simple))
f_simple['UniProt']=[0]*len(f_simple)
chunks = [f_simple.loc[f_simple.index[i:i + chunk_size]] for i in range(0, f_simple.shape[0], chunk_size)]
def func(df):
    mark=df.index[0]
    print('process %s starts....'%mark)
    for j,idx in enumerate(df.index):
        if 'UniProt' not in df.loc[idx,'OtherIDs']:
                df.loc[idx,'UniProt']=get_uniprot_from_name(f_simple.loc[idx,'Name'])
        else:df.loc[idx,'UniProt']=extract_uniprot(df.loc[idx,'OtherIDs'])
        if j%1000==0: 
            print ('\n\n\n\n %d completed \n\n\n\n'%j)
            # pbar.update(100)
    #TODO   df.to_csv('2022-2-remain_proc_%s_%s_finished.csv'%(mark,j))
    return df
pool = multiprocessing.Pool(processes=num_processes)
result =pool.map(func, chunks)
f_final=pd.concat(result,sort=False)
f_final.to_csv('/home/grads/z/zshuying/Documents/shuying/ppi_mutation/scripts/2022-02-variant-sum_remain_deal.csv')
# print(f_final)