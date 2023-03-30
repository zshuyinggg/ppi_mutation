from tqdm import tqdm
import multiprocessing
from os import path
import sys
from utils import *
import pandas as pd
import argparse
import requests
import json
import re

parser = argparse.ArgumentParser(description='Option to process clinvar data')
parser.add_argument('--if_raw_file', type=bool,default=False)
parser.add_argument('--input_file', type=str,required=True)
# parser.add_argument('--output_dir', type=str,required=True)
# parser.add_argument('--output_file_prefix', type=str,required=True)
parser.add_argument('--cpu_num',type=int,default=None)
# parser.add_argument('--review_filter',type=int,default=1, help='one star above or two star above for the review status')


args = parser.parse_args()
if_raw_file=args.if_raw_file
cpu_num=args.cpu_num
# review_filter=args.review_filter


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

def find_fasta_refseq(refseqID):
    url='https://rest.uniprot.org/uniprotkb/search?query=%s'%refseqID
    cnt=json.loads(requests.get(url).text)
    try:
        assert re.search('uniprot',cnt.get("results")[0].get("entryType"),re.IGNORECASE)
        # results=cnt["results"]
        # for i in range(len(results)):
        #     if (note in results[i]['uniProtkbId']) and cnt["results"][i]['primaryAccession']:
        #         return cnt["results"][i]['primaryAccession']
        #     elif (note in results[i]['uniProtkbId']) and (cnt["results"][i]['primaryAccession'] is None):
        #         return 'Primary Accession is None'
        return cnt["results"][0]['primaryAccession']
    except:
        print('Error found in '+url)
        return 'Error found in '+url

def get_uniprot_from_name(name):
    try:
        # refseq=re.match(r'\S*?(?=[\(:])',name).group(0)
        # note=name.split(':')[0].split('(')[1].split(')')[0]
        refseq=name.split(':')[0]
        uniprot=find_fasta_refseq(refseq)

    except:
        refseq='Error Finding Refseq'

        print('Error Finding Refseq')
        uniprot=find_fasta_refseq(refseq)
    return uniprot

def exlude_synonymous(name):
    #"NM_017813.5(BPNT2):c.228C>A (p.Arg76=)"
    return bool(re.match(r'.*(\(p\.[a-zA-Z0-9]*=\)).*',name))

def get_refseq_from_name(name):
    try:
        refseq=re.match(r'\S*?(?=[\(:])',name).group(0)

    except:
        refseq='Error Finding Refseq'
        print('Error Finding Refseq')
    return refseq

def func(df):
    mark=df.index[0]
    l=len(df.index)
    print('process %s starts....'%mark,flush=True)
    tmp_ref='asdfa'
    for j,idx in enumerate(df.index):
        current_ref=get_refseq_from_name(df.loc[idx,'Name'])
        if current_ref==tmp_ref:
            df.loc[idx,'UniProt']=tmp_uniprot
        elif (df.loc[idx,'UniProt']=='0' or df.loc[idx,'UniProt']==0 or pd.isna(df.loc[idx,'UniProt'])):
            df.loc[idx,'UniProt']=get_uniprot_from_name(df.loc[idx,'Name'])

        tmp_uniprot=df.loc[idx,'UniProt']
        tmp_ref=current_ref
        if j%500==0:
            print ('\n\n\n\n %d completed \n\n\n\n'%j,flush=True)
            df.to_csv('2020-6-round1_proc_%s_%s_finished.csv'%(mark,j))
            if j!=0:os.remove('2020-6-round1_proc_%s_%s_finished.csv'%(mark,j-500))
        if j==(l-1):
            df.to_csv('2020-6-round1_proc_%s_%s_finished.csv'%(mark,j))

    return df

# get_uniprot_from_name('NM_000512.5(GALNS):c.1171A>G (p.Met391Val)')
#

if __name__ == '__main__':
    if if_raw_file:

        f=pd.read_csv(args.input_file,sep='\t')
        f['label']=if_positive_or_negative(f['ClinicalSignificance'])
        print(f.head(5))
        keep_cols=['#AlleleID', 'Type', 'Name','label', 'ClinicalSignificance', 'ClinSigSimple', 'ReviewStatus', 'OtherIDs', 'LastEvaluated', 'RS# (dbSNP)',
            'nsv/esv (dbVar)', 'RCVaccession']
        keep_conditions=(f['Type']=='single nucleotide variant') &\
                    (['no assertion' not in item for item in f['ReviewStatus'].tolist()]) &\
                    (['p' in str(item) for item in f['Name'].tolist()]) &\
                    (f['label']!=0) &\
                    ([bool(exlude_synonymous(str(name)))==False for name in f['Name'].tolist()])
        f_simple=f[keep_conditions][keep_cols]
        f_simple=f_simple.drop_duplicates(subset=['#AlleleID'])
        f_simple['UniProt']=[0]*len(f_simple)
        print('The data has %s rows'%len(f_simple.index))

    else:
        f_simple=pd.read_csv(args.input_file,index_col='#AlleleID',dtype=str)

        f_simple[(f_simple['UniProt']!='0') & (f_simple['UniProt'].notnull())].to_csv('2020-6-round1_done.csv')
        f_simple=f_simple[(f_simple['UniProt']=='0') | (f_simple['UniProt'].isnull())]
    if cpu_num:
        num_processes=cpu_num
    else:
        num_processes = multiprocessing.cpu_count()
    print('%s cpus'%num_processes)

    # f_simple=pd.read_csv('/home/grads/z/zshuying/Documents/shuying/ppi_mutation/scripts/merged_2023_2_26_round4.csv',index_col='#AlleleID')

    # f_simple[f_simple['UniProt']!='0'].to_csv('2023-2-round5_proc_already_done_finished.csv')
    # f_simple=f_simple[f_simple['UniProt']=='0']

    if cpu_num==1: func(f_simple)
    else:

        chunk_size = int(f_simple.shape[0]/num_processes)
        print(chunk_size)
        chunks = [f_simple.loc[f_simple.index[i:i + chunk_size]] for i in range(0, f_simple.shape[0], chunk_size)]

        pool = multiprocessing.Pool(processes=num_processes)
        result =pool.map(func, chunks)
        f_final=pd.concat(result,sort=False)
        f_final.to_csv('/home/grads/z/zshuying/Documents/shuying/ppi_mutation/scripts/merged_2022_2_round2.csv')
        print(f_final)