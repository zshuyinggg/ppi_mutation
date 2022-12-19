
#%%

import csv
import xlrd
import pandas as pd
from tqdm import tqdm
import requests
import numpy as np

import time
import json
import zlib
from xml.etree import ElementTree
from urllib.parse import urlparse, parse_qs, urlencode
import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup as BS
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles
import json
import os, sys
global top_path  # the path of the top_level directory
global data_dir, script_dir, logging_dir
# add the top-level directory of this project to sys.path so that we can import modules without error
POLLING_INTERVAL = 3
API_URL = "https://rest.uniprot.org"
retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
session = requests.Session()
def find_current_path():
    if getattr(sys, 'frozen', False):
        # The application is frozen
        current = sys.executable
    else:
        # The application is not frozen
        # Change this bit to match where you store your data files:
        current = __file__

    return current

top_path = os.path.dirname(os.path.dirname(os.path.realpath(find_current_path())))
sys.path.append(top_path)
def get_cli_significance(snpid):
    response = requests.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=snp&rettype=vcv&id=%s&from_esearch=true'%snpid).text
    soup=BS(response,'xml')
    clinical_significance=soup.find('CLINICAL_SIGNIFICANCE')

    try:
        print(clinical_significance.text)
        return clinical_significance.text
    except AttributeError:
        print('snpid',snpid,clinical_significance)

def check_response(response):
    try:
        response.raise_for_status()
    except requests.HTTPError:
        print(response.json())
        raise

def get_id_mapping_results_link(job_id):
    url = f"{API_URL}/idmapping/details/{job_id}"
    request = session.get(url)
    check_response(request)
    return request.json()["redirectURL"]

def get_uniprotid_from_refseq(accessID):

    session.mount("https://", HTTPAdapter(max_retries=retries))
    request = requests.post(
        f"{API_URL}/idmapping/run",
        data={"from": 'RefSeq_Protein', "to": 'UniProtKB', "ids": '%s'%accessID},
    )
    check_response(request)
    jobid=request.json()["jobId"]
    result_url=get_id_mapping_results_link(jobid)
    response = requests.get(result_url)
    soup = BS(response.text,'html.parser')
    site_json=json.loads(soup.text)
    uniprot_id=site_json['results'][0]['to']['primaryAccession']
    print(uniprot_id)




def add_indents(txt,out):
    import csv
    from tqdm import tqdm

    with open(txt,'r') as f_in:
        num_lines = sum(1 for line in f_in)
    with open(txt,'r') as f_in:
        flag=1
        with open(out,'w',newline='') as f_out:
            for i in tqdm(range(num_lines)):
                row=f_in.readline()
                if flag: 
                    s0=row.count('\t')
                    flag=0

                else: 
                    s=row.count('\t')
                    row=row.strip('\n')+(s0-s)*'\t'+'\n'
                    # row=row.replace('\t', ',')

                f_out.write(row)
    
    print('%s has been converted to %s'%(txt,out))


def process_variants(data,out):
    # '/home/grads/z/zshuying/Documents/shuying/ppi_mutation/variant_summary_indents.txt'
    map_dict={
        'Type':1,
        'Name':2,
        'GeneID':3,
        'GeneSymbol':4,
        'ClinicalSignificance':6,
        'ClinSigSimple':7,
        'LastEvaluated':8,
        'RS# (dbSNP)':9,
        'ReviewStatus':10
    }

    with open(data,'r') as f_in:
        num_lines = sum(1 for line in f_in)
    with open(data,'r') as f_in:

        # get the mappings
        header=f_in.readline().strip('\n').split('\t')
        keep=[]
        ct=0
        for i,column in enumerate(header):
            if map_dict.get(column):
                map_dict[column]=i
                ct+=1
                keep.append(i)
        assert ct==9
        with open(out,'w') as f_out:
            for i in keep:
                f_out.write(header[i])
            f_out.write('\n')
            for i in tqdm(range(num_lines)):
                row=f_in.readline().split('\t')
                if row[map_dict['Type']]!=''
                for i in keep:
                    f_out.write(row[i])
                f_out.write('\n')



    
def get_uniprot_variant_summary(snpid):
    response = requests.get(
        'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=snp&rettype=vcv&id=%s&from_esearch=true' % snpid).text
    soup = BS(response, 'xml')
    HGVS=soup.find('DOCSUM')

    HGVS_p=list(map(lambda x: 'NP'+x, re.split('NP',HGVS)))

    # get uniprot id
    uniprot_id=get_uniprotid_from_refseq(HGVS_p)

    #
    # # return review status
    # HGVS_g=re.split(',',HGVS)
    # for item in HGVS_g:
    #     if 'g.' not in item:
    #         HGVS_g.remove(item)
    # # pick one hgvs_g #TODO: check all of them and pick the one with expert reviewed clinical importance
    # HGVS_g=HGVS_g[0]
    #
    #
    # # get clinical importance
    # clinical_importance=get_cli_significance(snpid)


#%%
import xml.etree.ElementTree as ET
from pandas.errors import ParserError
#get an iterable
file='/home/grads/z/zshuying/Documents/shuying/ppi_mutation/data/clinvar/ClinVarFullRelease_00-latest.xml'
out='/home/grads/z/zshuying/Documents/shuying/ppi_mutation/data/clinvar/organize_clinar.txt'
context= ET.iterparse(file,events=('start','end'))


# get the line count of the file:
with open(file,'r') as f:

    try:
        num_lines = 0
        for i in context:
            num_lines+=1    
            del i
    except SyntaxError: pass

#%%

from tqdm import tqdm
context= ET.iterparse(file,events=('start','end'))

node=0
node_measure=0
pbar=tqdm(total=num_lines)
clinvar_id,review_status,clinical_sig,uniprot_kb,variant_type,hgvs_p,missense=0,0,0,0,0,0,0

with open(out,'w') as f_out:
    f_out.write('clinvar_id,review_status,clinical_sig,clinvar_id,uniprot_kb,variant_type,hgvs_p,missense\n')
    try:
        for index, (event,elem) in enumerate(context):
            pbar.update(1)
            write_flag=1
            #get the root element
            if index==0:
                root=elem
                elem.clear()
            if event=='start' and elem.tag=='ClinicalSignificance':
                node=1
                elem.clear()
            if event=='end' and elem.tag=='ClinicalSignificance':
                node=0
                elem.clear()
            if event=='start' and elem.tag=='ClinVarAccession':
                clinvar_id=elem.attrib['Acc']
                elem.clear()
            if event=='end' and elem.tag=='ReviewStatus' and node:
                review_status=elem.text
                elem.clear()
            if event=='end' and elem.tag=='Description' and node:
                clinical_sig=elem.text
                elem.clear()
            if event=='end' and elem.tag=='XRef' and elem.attrib['DB']=='UniProtKB':
                uniprot_kb=elem.attrib['ID']
                elem.clear()
            if event=='start' and elem.tag=='MeasureSet':
                node_measure=1
                elem.clear()
            if event=='end' and elem.tag=='MeasureSet':
                node_measure=0
                elem.clear()
            if node_measure and event=='end' :
                if elem.tag=='Measure':
                    # print (elem.attrib['Type'])
                    # if variant_type!='single nucleotide variant':continue #some SNV are marked as "variation"
                    variant_type=elem.attrib['Type']
                    elem.clear()
                if elem.tag=='Attribute':
                    if elem.attrib['Type']=='HGVS, protein':
                        print(elem.text)
                        hgvs_p=elem.text
                    if elem.attrib['Type']=='MolecularConsequence':
                        missense=elem.text
                        elem.clear()
                        if missense!='missense variant':
                            continue
            if 0 in [clinvar_id,review_status,clinical_sig,variant_type,uniprot_kb,missense]: continue # if the record misses the feature we need then skip it
            try:f_out.write(','.join([clinvar_id,review_status,clinical_sig,uniprot_kb,variant_type,hgvs_p,missense])+'\n')
            except TypeError:
                print([clinvar_id,review_status,clinical_sig,uniprot_kb,variant_type,hgvs_p,missense])
            clinvar_id,review_status,clinical_sig,uniprot_kb,variant_type,hgvs_p,missense=0,0,0,0,0,0,0
        pbar.close()
    except SyntaxError: pass








#%%
get_uniprotid_from_refseq('NP_000228.1')
# %%
convert_txt_csv('/home/grads/z/zshuying/Documents/shuying/ppi_mutation/variant_summary.txt','/home/grads/z/zshuying/Documents/shuying/ppi_mutation/variant_summary_indents.txt')
# %%

# too big to query
convert_txt_csv('/home/grads/z/zshuying/Documents/shuying/ppi_mutation/hgvs4variation.txt','/home/grads/z/zshuying/Documents/shuying/ppi_mutation/hgvs4variation_indents.txt')
# %%

import numpy as np
variants=np.loadtxt('/home/grads/z/zshuying/Documents/shuying/ppi_mutation/hgvs4variation_indents.txt',delimiter='\t',dtype=str)
# %%
