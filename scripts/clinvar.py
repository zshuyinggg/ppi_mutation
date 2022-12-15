
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
    
def get_uniprot_variant(snpid):
    response = requests.get(
        'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=snp&rettype=vcv&id=%s&from_esearch=true' % snpid).text
    soup = BS(response, 'xml')
    HGVS=soup.find('DOCSUM')

    HGVS_p=list(map(lambda x: 'NP'+x, re.split('NP',HGVS)))

    # get uniprot id
    uniprot_id=get_uniprot_variant(HGVS_p)


    # return review status
    HGVS_g=re.split(',',HGVS)
    for item in HGVS_g:
        if 'g.' not in item:
            HGVS_g.remove(item)
    # pick one hgvs_g #TODO: check all of them and pick the one with expert reviewed clinical importance
    HGVS_g=HGVS_g[0]


    # get clinical importance
    clinical_importance=get_cli_significance(snpid)



#%%
get_uniprotid_from_refseq('NP_000228.1')
# %%
