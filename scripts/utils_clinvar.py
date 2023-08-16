import configparser
#%%
import csv
import xlrd
import pandas as pd
from tqdm import tqdm
import math
import requests
import numpy as np
from bs4 import BeautifulSoup as BS
from loguru import logger

import re
import pandas as pd
from Bio.SeqUtils import seq1
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles,venn3_unweighted
import json
import os, sys
global top_path  # the path of the top_level directory
global script_path, data_path, logging_path
# from biotransformers import BioTransformers
# add the top-level directory of this project to sys.path so that we can import modules without error
pj=os.path.join
top_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(top_path)
script_path, data_path, logging_path= pj(top_path,'scripts'), \
    pj(top_path,'data'), \
    pj(top_path,'logs')



class MultiProcessClinvar():
    list_dfs=[]
    def __init__(self,variant_summary_file=None,df=None,review_status=1,ref_uniprot_file=pj(data_path,'single_protein_seq/uniprotids_humap_huri.txt')) -> None:
        """
        if variant_summary_file is provided, then this instance is used to preprocess the variant_summary_file to a dataframe,
        if df is provide, then this instance is used to multiprocess the dataframe (get the uniprot and seqs)
        """
        self.all_ppi_uniprot_ids = eval(open(ref_uniprot_file).readline()) 
        if df is None:
            print("Preprocess variant summary file %s"%variant_summary_file)
            self.df=self.preprocess(variant_summary_file,review_status)
            print("Preprocess variant summary file done with review status = %s"%(review_status))

        else:
            # print('df is not none',df)
            self.df=df 
            print('subprocess %s starts, with items total=%s'%(os.getpid(),len(df)))
    def process_variants_df(self,save_name):
        pid=os.getpid()
        tqdm_text = "#" + "{}".format(pid).zfill(3)
        cached_refseq,cached_uniprot,cached_seq=None,None,None
        with tqdm(total=len(self.df), desc=tqdm_text, position=pid+1) as pbar:
            for i,idx in enumerate(self.df.index): 
                current_refseq=get_refseq_from_name(self.df.loc[idx,'Name'])
                if current_refseq==cached_refseq:
                    self.df.loc[idx,'UniProt']=cached_uniprot
                    self.df.loc[idx,'Seq']=modify(cached_seq,self.df.loc[idx,'Name'])
                else:
                    self.df.loc[idx,'UniProt'],wild_seq=get_uniprot_seq_from_refseq(current_refseq)
                    self.df.loc[idx,'Seq']=modify(wild_seq,self.df.loc[idx,'Name'])
                    cached_refseq,cached_uniprot,cached_seq=current_refseq,self.df.loc[idx,'UniProt'],wild_seq
                pbar.update(1)
        # all uniprot ids that involve in PPI of humap and huri
        self.df = self.df[[uniprot in self.all_ppi_uniprot_ids for uniprot in self.df['UniProt'].tolist()]]
        #only include variants that are in PPI
        # self.df.to_csv('%s_%s.csv'%(save_name,pid))
        return self.df
    def preprocess(self,variant_summary_file,review_status):
        f=pd.read_csv(variant_summary_file,sep='\t')
        f['Label']=if_positive_or_negative(f['ClinicalSignificance'])
        keep_cols=['#AlleleID', 'Type', 'Name','Label', 'ClinicalSignificance', 'ClinSigSimple', 'ReviewStatus', 'OtherIDs', 'LastEvaluated', 'RS# (dbSNP)',
        'nsv/esv (dbVar)', 'RCVaccession']
        keep_conditions=(f['Type']=='single nucleotide variant') &\
                (['no assertion' not in item for item in f['ReviewStatus'].tolist()]) &\
                (['p' in str(item) for item in f['Name'].tolist()]) &\
                (f['Label']!=0) &\
                ([bool(exlude_synonymous(str(name)))==False for name in f['Name'].tolist()]) &\
                  (boolean_review_status(f['ReviewStatus'],review_status))
        f_simple=f[keep_conditions][keep_cols]
        f_simple=f_simple.drop_duplicates(subset=['#AlleleID'])
        f_simple['UniProt']=[0]*len(f_simple)
        f_simple['Seq']=[0]*len(f_simple)
        f_simple.reset_index(drop=True)
        print(f.head(5))
        print('Total: %s rows'%len(f_simple.index))
        return f_simple




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

def exlude_synonymous(name):
    #"NM_017813.5(BPNT2):c.228C>A (p.Arg76=)"
    return bool(re.match(r'.*(\(p\.[a-zA-Z0-9]*=\)).*',name))



def get_refseq_from_name(name):
    try:
        refseq=name.split(':')[0]

    except:
        refseq='Error Finding Refseq from the Name in variant summary file'
        print('Error Finding Refseq from the Name in variant summary file')
    return refseq


def get_uniprot_seq_from_refseq(refseqID):
    """
    return uniprot and seq
    """
    url='https://rest.uniprot.org/uniprotkb/search?query=%s'%refseqID #e.g.https://rest.uniprot.org/uniprotkb/search?query=NM_000162.5(GCK)
    cnt=json.loads(requests.get(url).text)
    #get uniprot ID
    try:
        assert re.match('.*uniprot.*',cnt.get("results")[0].get("entryType"),re.IGNORECASE)
        uniprot= cnt["results"][0]['primaryAccession']
    except:
        print('Error getting UniProt ID in '+url)
        uniprot= 'Error getting UniProt ID in '+url
        return uniprot,uniprot
    try:
        assert cnt.get("results")[0].get("sequence").get("value")
        seq= cnt["results"][0]['sequence']["value"]
    except:
        print('Error getting seq in '+url)
        seq= 'Error getting seq in '+url
    return uniprot,seq

def get_loc_from_name(name):
    change=name.split('p.')[1]
    obj=re.findall(r'[0-9]+',change)[0]
    return int(obj)-1 
def modify(seq,hgvs):
    #NP_000240.1:p.Ile219Val
    change=hgvs.split('p.')[1]
    obj=re.match(r'([a-zA-Z]+)([0-9]+)([a-zA-Z]+)',change)
    if '=' in change: #NM_000238.4(KCNH2):c.1539C>T (p.Phe513_Gly514=)	
        obj=re.match(r'([a-zA-Z]+)([0-9]+)',change.split('_')[0])
        try:
            ori,pos=obj.group(1),int(obj.group(2))
            assert ori==seq[pos-1]
        except:
            new_seq='Error!! !!!' 
            return new_seq
        return seq[:pos-1]+seq[pos:]
    elif obj is None:
        print('%s did not find match'%hgvs)
        new_seq='Error!! did not find match'
        return new_seq
    ori,pos,aft=obj.group(1),int(obj.group(2)),obj.group(3)
    ori=seq1(ori)
    if ori == '*': seq=seq+'*' # if the changes is made to the terminator, then the seq does not have it, we have to add it first
    aft=seq1(aft) 
    try:assert ori==seq[pos-1]
    except (AssertionError,IndexError):
        new_seq='Error!! Isoform is probably wrong!!!' #TODO: edit the code to deal with isoform
        return new_seq

    new_seq=seq[:pos-1] + aft + seq[pos:]
    try:assert new_seq[pos-1]==aft

    except (AssertionError,IndexError):
        new_seq='Error!! Isoform is probably wrong!!!' #TODO: edit the code to deal with isoform
        return new_seq
    if aft=='*': new_seq=new_seq.split('*')[0] #cut the seq according to terminator
    return new_seq


