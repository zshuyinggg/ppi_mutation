import csv
import xlrd
import pandas as pd
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
    dic=pd.read_csv('../data/string_uniprot_lookup.csv')




