import csv
import xlrd
import pandas as pd
from tqdm import tqdm
import requests
import numpy as np
from bs4 import BeautifulSoup as BS

#'../data/human_ppi_9606.protein.links.full.v11.5.stringdb.txt'
def get_stringdb_id(data):
    # map the returned file at https://www.uniprot.org/tool-dashboard
    l=[]
    with open(data) as f:
        reader=csv.reader(f,delimiter=' ',quotechar='|')
        for row in reader:
            l.append(row[0])
            l.append(row[1])
        s=set(l)
        with open('../data/stringIDs.txt','w') as f:
            for item in s:
                f. write('%s,'%item)
    return



def map_stringdb_uniprot():
    f=pd.read_csv('../data/string_uniprot_lookup.csv')
    ppi_stringdb=pd.read_csv('../data/human_ppi_9606.protein.links.full.v11.5.stringdb.txt',sep=" ")
    dic=dict(zip(f['From'],f['Entry']))
    ppi_uniprot=ppi_stringdb.copy()
    j=0
    flag=0
    for i in tqdm(range(len(ppi_stringdb))):
        if i+j<len(ppi_uniprot):
            try:
                ppi_uniprot.iloc[i, 0] = dic[ppi_stringdb.iloc[i+j, 0]]
                ppi_uniprot.iloc[i,1]=dic[ppi_stringdb.iloc[i+j,1]]
            except(KeyError):
                # print('%s not found in uniprot '%ppi_stringdb.iloc[i,1])
                j+=1
                continue
        else:
            flag=i
            break
    print('%s rows are deleted'%j)
    ppi_out=ppi_uniprot[['protein1','protein2','experiments','experiments_transferred']].iloc[:flag,:]
    print(len(ppi_out))
    print(len(ppi_stringdb))
    print(ppi_out.head())
    ppi_out.to_csv('../data/ppi_uniprot.csv',index=False)

def get_cli_significance(snpid):
    response = requests.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=snp&rettype=vcv&id=%s&from_esearch=true'%snpid).text
    soup=BS(response,'xml')
    clinical_significance=soup.find('CLINICAL_SIGNIFICANCE')

    try:
        print(clinical_significance.text)
        return clinical_significance.text
    except AttributeError:
        print('snpid',snpid,clinical_significance)


def add_cli_significance_to_file(file_in,file_out):
    f=pd.read_csv(file_in)
    print(f.columns)
    list_snps=list(set(f.loc[:,'dbSNP_id']))
    table=pd.DataFrame(columns=['dbSNP_id','clinical_significance'],index=list_snps)
    table['clinical_significance']=np.zeros((len(table),1))
    # if table.columns[-1]!='clinical_significance' :
    #     raise NameError("the last column should be 'clinical_significance'")
    print(table.head())
    for i in list_snps:
        table.loc[i,'clinical_significance']=get_cli_significance(i)
    table.to_csv(file_out)

get_cli_significance('rs328')
add_cli_significance_to_file('../data/supplementary2.csv','../data/supplementary2_significance.csv')
#
# add_cli_significance_to_file('../data/supplementary3.csv','../data/supplementary3_significance.csv')
# add_cli_significance_to_file('../data/supplementary4.csv','../data/supplementary4_significance.csv')
