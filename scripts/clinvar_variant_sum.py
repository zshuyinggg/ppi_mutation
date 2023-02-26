from tqdm import tqdm
import multiprocessing
from os import path
import sys
from utils import *

import pandas as pd
f=pd.read_csv(os.path.join(data_dir,'clinvar','variant_summary.txt'),sep='\t')
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

# f_simple.head()


num_processes = multiprocessing.cpu_count()
print('%s gpus'%num_processes)
chunk_size = int(f_simple.shape[0]/num_processes)
global pbar
pbar = tqdm(total=len(f_simple))
f_simple['UniPort']=[0]*len(f_simple)
chunks = [f_simple.loc[f_simple.index[i:i + chunk_size]] for i in range(0, f_simple.shape[0], chunk_size)]
def func(df):
    mark=df.index[0]
    for j,idx in enumerate(df.index):
        if 'UniProt' not in df.loc[idx,'OtherIDs']:
                df.loc[idx,'UniPort']=get_uniprot_from_name(f_simple.loc[idx,'Name'])
        if j%1000==0: 
            print ('\n\n\n\n %d completed \n\n\n\n'%idx)
            pbar.update(1000)
            df.to_csv('variant_proc_%s_%s_finished.csv'%(mark,j))
    return df
pool = multiprocessing.Pool(processes=num_processes)
result =pool.map(func, chunks)
f_final=pd.concat(result,sort=False)
f_final.to_csv('variant_sum_uniprot.csv')
print(f_final)
