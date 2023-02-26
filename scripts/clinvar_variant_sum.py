from tqdm import tqdm
import multiprocessing


import pandas as pd
f=pd.read_csv('../data/clinvar/variant_summary.txt',sep='\t')
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
f_simple.head()


num_processes = multiprocessing.cpu_count()
print('%s gpus'%num_processes)
chunk_size = int(f_simple.shape[0]/num_processes)
global pbar
pbar = tqdm(total=len(f_simple))
f_simple['UniPort']=[0]*len(f_simple)
chunks = [f_simple.loc[f_simple.index[i:i + chunk_size]] for i in range(0, f_simple.shape[0], chunk_size)]
def func(df):
    for idx in df.index:
        if 'UniProt' not in df.loc[idx,'OtherIDs']:
                df.loc[idx,'UniPort']=get_uniprot_from_name(f_simple.loc[idx,'Name'])
    return df
pool = multiprocessing.Pool(processes=num_processes)
result =list(tqdm(pool.imap(func, chunks),total=len(chunk_size)))
