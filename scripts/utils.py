import csv
import xlrd
import pandas as pd
from tqdm import tqdm
import requests
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

def get_clinvar_page(snpid):
    import requests


    response = requests.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=clinvar&rettype=vcv&is_variationid&id=rs3829740&from_esearch=true').text
    # page=requests.get('https://www.ncbi.nlm.nih.gov/snp/'+str(snpid)).text
    # soup=BS(response,'html.parser')

    print(response)


get_clinvar_page('rs3829740')

# map_stringdb_uniprot()