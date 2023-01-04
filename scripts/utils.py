
#%%
import csv
import xlrd
import pandas as pd
from tqdm import tqdm
import requests
import numpy as np
from bs4 import BeautifulSoup as BS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles,venn3_unweighted
import json
import os, sys
global top_path  # the path of the top_level directory
global data_dir, script_dir, logging_dir
# add the top-level directory of this project to sys.path so that we can import modules without error

top_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(top_path)
def get_local_paths():
    with open(file=os.path.join(top_path, 'scripts/path.json')) as f:
        mydict = json.load(f)
    return mydict['script_path'], mydict['data_path'], mydict['logging_path']


script_path, data_path, logging_path = get_local_paths()
  
def get_stringdb_id(data):
#'../data/9606.protein.physical.links.detailed.v11.5.txt'

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
    f=pd.read_csv(os.path.join(data_path,'string_uniprot_lookup.csv'))
    ppi_stringdb=pd.read_csv(os.path.join(data_path,'9606.protein.physical.links.detailed.v11.5.txt'),sep=" ")
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
                i-=1
                continue
        else:
            flag=i
            break
    print('%s rows are deleted'%j)
    ppi_out=ppi_uniprot[['protein1','protein2','combined_score']].iloc[:flag,:]
    print(len(ppi_out))
    print(len(ppi_stringdb))
    print(ppi_out.head())
    ppi_out.to_csv(os.path.join(data_path,'string_ppi_uniprot.csv'),index=False)

def screen_by_confidence(data,cutoff):
    f=pd.read_csv(data)
    f=f[f.combined_score>=cutoff]
    print(len(f))
    f.to_csv('%s_screened_%f'%(data,cutoff))
# screen_by_confidence(os.path.join(data_path,'string_ppi_uniprot.csv'),400)
# screen_by_confidence(os.path.join(data_path,'string_ppi_uniprot.csv'),700)
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

def map_huRI_uniprot():
    f=pd.read_csv('../data/complexes/huintaf2-main/data/HuRI-merged-allnames.csv')
    return f


def get_all_uniprot_3_databases():
    humap=set(pd.read_csv(os.path.join(data_path,'complexes/huintaf2-main/data/HuMap-uniprot.tab'),sep='\t')['Entry'].tolist())
    with open(os.path.join(data_path,'complexes/huintaf2-main/data/HuRI-uniprot-ids.txt'),'r') as f2:
        huri=set(f2.read().splitlines())
    string=set(pd.read_csv(os.path.join(data_path,'string_uniprot_lookup.csv'))['Entry'].tolist())
    venn3([humap, huri, string],
          set_colors=('#3E64AF', '#3EAF5D', '#D74E3B'),
          set_labels=('humap',
                      'huri',
                      'string'),
          alpha=0.75)
    venn3_circles([humap, huri, string], lw=0.7)
    plt.show()

def compare_ppi_3_databases():
    humap=pd.read_csv(os.path.join(data_path,'complexes/huintaf2-main/data/humap_pairs.csv'),names=['name','p1','p2']).iloc[:,1:]
    huri=pd.read_csv(os.path.join(data_path,'complexes/huintaf2-main/data/HuRI-merged-allnames.csv'),header=0).loc[:,['id1','id2']]
    # print(huri.head())
    string=pd.read_csv(os.path.join(data_path,'string_ppi_uniprot.csv')).loc[:,['protein1','protein2']]
    humap['pair1']=humap['p1']+'-'+humap['p2']
    humap['pair2']=humap['p2']+'-'+humap['p1']
    huri['pair1']=huri['id1']+'-'+huri['id2']
    huri['pair2']=huri['id2']+'-'+huri['id1']
    string['pair1']=string['protein1']+'-'+string['protein2']
    string['pair2']=string['protein2']+'-'+string['protein1']
    humap_set=set(humap['pair1'].tolist()).union(set(humap['pair2'].tolist()))
    huri_set=set(huri['pair1'].tolist()).union(set(huri['pair2'].tolist()))
    string_set=set(string['pair1'].tolist()).union(set(string['pair2'].tolist()))
    venn3([humap_set, huri_set, string_set],
          set_colors=('#3E64AF', '#3EAF5D', '#D74E3B'),
          set_labels=('humap',
                      'huri',
                      'string'),
          alpha=0.75)
    venn3_circles([humap_set, huri_set, string_set], lw=0.7)
    plt.show()


def convert_huri_ensembl_2_uniprot(l):
    """

    :param l: list of strings which are the ensembl ID of  the protein pairs. Example:ENSG00000000005-ENSG00000061656
    :return:  uniprot ID pairs
    """
    # get the dictionary
    f=pd.read_csv(os.path.join(data_path,'complexes/huintaf2-main/data/HuRI-merged-allnames.csv'),header=0)
    dic=dict(zip(f.EnsName,f.Name))
    values=[]
    for key in l:
        try:values.append(dic[key])
        except KeyError: print('pair name %s found and ignored'%key)
    return values

def generate_set_of_pairs(l1,if_self_included):
    l1=[item for item in l1 if '.' not in item and len(item.split('-'))==2]
    # l11=[item.split('-')[0] for item in l1 if len(item.split('-'))==2]
    # l12=[item.split('-')[1] for item in l1 if len(item.split('-'))==2]
    l_self=[item for item in l1 if (len(item.split('-'))==2) and (item.split('-')[0]==item.split('-')[1])]
    if '-' in l_self:l_self.remove('-')
    if '-' in l1:l1.remove('-')
    l_self_duplicate=[item.split('-')[0]+'_dup_'+item.split('-')[1] for item in l_self] #This is to count the self-interacting pairs
    l_error=[item for item in l1 if len(item.split('-'))<2]
    print(l_error)
    for item in l_error:
        l1.remove(item)
    from functools import partial, reduce
    l1_reverse=[item.split('-')[1]+'-'+item.split('-')[0] for item in l1 if len(item.split('-'))==2]
    if '-' in l1_reverse:l1_reverse.remove('-')

    set1=set(l1)
    set1.update(l1_reverse)
    if if_self_included:
        set1.update(l_self_duplicate)
    else:
        set1=set1-set(l_self)
    return set1

def get_self_interaction(l1):
    set1=set(l1)
    l11=[item.split('-')[0] for item in l1 if len(item.split('-'))>1]
    l12=[item.split('-')[1] for item in l1 if len(item.split('-'))>1]
    l_self=[item for item in l1 if len(item.split('-'))>1 and item.split('-')[0]==item.split('-')[1]]
    # l_error=[item for item in l1 if len(item.split('-'))<2]
    # print(l_error)
    # from functools import partial, reduce
    # l1_reverse=list(reduce(partial(map, str.__add__), (l12,'-', l11)))
    # set2=set(l1_reverse)
    return l_self
def compare_actual_ppi_3_databases(p,if_self_included):
    """
    Compare screened three databases (Huri and humap i just read their actual used ppi from the directory)
    :param p: the cutoff of confidence of stringdb
    :return: venn graph
    """
    huri_dir=os.path.join(data_path,'complexes/HuRI') #Ensembl pairs
    humap_dir=os.path.join(data_path,'complexes/pdb') #uniprot pairs
    huri_uniprot_pairs=convert_huri_ensembl_2_uniprot([x[0].split('/')[-1] for x in os.walk(huri_dir)])
    humap_uniprot_pairs=[x[0].split('/')[-1] for x in os.walk(humap_dir)]
    huri_uniprot_pairs=generate_set_of_pairs(huri_uniprot_pairs,if_self_included)
    humap_uniprot_pairs=generate_set_of_pairs(humap_uniprot_pairs,if_self_included)

    f=pd.read_csv(os.path.join(data_path,'string_ppi_uniprot.csv'))
    print('before screening, StringDB has %s'%len(f))
    f=f[f.combined_score>=p]
    print('after screening, StringDB has %s'%len(f))

    f['pair1']=f['protein1']+'-'+f['protein2']
    f['pair2']=f['protein2']+'-'+f['protein1']
    string_set=set(f['pair1'].tolist()).union(set(f['pair2'].tolist()))
    venn3_unweighted([humap_uniprot_pairs, huri_uniprot_pairs, string_set],
          set_colors=('#3E64AF', '#3EAF5D', '#D74E3B'),
          set_labels=('humap',
                      'huri',
                      'string'),
          alpha=0.75)
    # venn3_circles([humap_uniprot_pairs, huri_uniprot_pairs, string_set], lw=0.3)
    plt.show()
    print(huri_uniprot_pairs.intersection(humap_uniprot_pairs))


def get_uniprot_ids(list_of_databases):
    """list of 'huri' and/or 'humap'
    """
    ids=set()
    for database in list_of_databases:
        print(database)
        if database=='huri':
            huri_set=set()

            huri_dir=os.path.join(data_path,'complexes/HuRI') #Ensembl pairs
            huri_uniprot_pairs=convert_huri_ensembl_2_uniprot([x[0].split('/')[-1] for x in os.walk(huri_dir)])
            l=[item.split('-') for item in huri_uniprot_pairs if '.' not in item and len(item.split('-'))==2]
            # print(l)
            if '-' in l:l.remove('-')

            for item in l:

                huri_set=huri_set.union(set(item))
            print('huri length:', len(huri_set))
            
        elif database=='humap':
            humap_set=set()
            humap_dir=os.path.join(data_path,'complexes/pdb') #uniprot pairs
            humap_uniprot_pairs=[x[0].split('/')[-1] for x in os.walk(humap_dir)]
            l=[item.split('-') for item in humap_uniprot_pairs if '.' not in item and len(item.split('-'))==2]
            # print(l)
            if '-' in l:l.remove('-')
            for item in l:
                humap_set=humap_set.union(set(item))

            print('humap length:', len(humap_set))

        else: raise NameError('%s is not a right argument'%list_of_databases)
    ids=humap_set.union(huri_set)
    if '' in ids: ids.remove('')
    return ids

def find_pairs(string):
    huri_dir=os.path.join(data_path,'complexes/HuRI') #Ensembl pairs
    humap_dir=os.path.join(data_path,'complexes/pdb') #uniprot pairs
    huri_uniprot_pairs=convert_huri_ensembl_2_uniprot([x[0].split('/')[-1] for x in os.walk(huri_dir)])
    humap_uniprot_pairs=[x[0].split('/')[-1] for x in os.walk(humap_dir)]
    huri_uniprot_pairs=generate_set_of_pairs(huri_uniprot_pairs,1)
    humap_uniprot_pairs=generate_set_of_pairs(humap_uniprot_pairs,1)
    for item in huri_uniprot_pairs:
        if string in item:
            print('string %s is in huri pair %s'%(string,item))
    for item in humap_uniprot_pairs:
        if string in item:
            print('string %s is in humap pair %s'%(string,item))




def get_sequence_from_uniprot_id(id):
    url='https://rest.uniprot.org/uniprotkb/%s.fasta'%id
    

def summary_self_interactions(p):
    """
    check if any of the pairs are self-self interaction
    """
    huri_dir=os.path.join(data_path,'complexes/HuRI') #Ensembl pairs
    humap_dir=os.path.join(data_path,'complexes/pdb') #uniprot pairs
    huri_uniprot_pairs=convert_huri_ensembl_2_uniprot(os.listdir(huri_dir)[1:])
    humap_uniprot_pairs=os.listdir(humap_dir)[1:]

    huri_self_pair_count=len(get_self_interaction(huri_uniprot_pairs))
    humap_self_pair_count=len(get_self_interaction(humap_uniprot_pairs))
    f=pd.read_csv(os.path.join(data_path,'string_ppi_uniprot.csv'))
    print('before screening, StringDB has %s'%len(f))
    f=f[f.combined_score>=p]
    print('after screening, StringDB has %s'%len(f))

    f['pair1']=f['protein1']+'-'+f['protein2']
    f['pair2']=f['protein2']+'-'+f['protein1']
    string_self_pair_count=sum(f['pair1']==f['pair2'])
    print('huri self interacting pairs:', huri_self_pair_count)
    print('humap self interacting pairs:', humap_self_pair_count)
    print('string self interacting pairs:', string_self_pair_count)


def screen_clinvar():
    import allel
    df = allel.vcf_to_dataframe(os.path.join(data_path,'clinvar','clinvar.vcf'))
    df.head()





# import matplotlib
# matplotlib.use('macosx')
# %% 
# get_all_uniprot_3_databases()
# compare_actual_ppi_3_databases(400)
# get_stringdb_id('../data/9606.protein.physical.links.detailed.v11.5.txt')
# # map_stringdb_uniprot()
# # screen_by_confidence('../data/ppi_uniprot.csv',400)
# get_cli_significance('rs328')
# # add_cli_significance_to_file('../data/supplementary2.csv','../data/supplementary2_significance.csv')
# # #
# # add_cli_significance_to_file('../data/supplementary3.csv','../data/supplementary3_significance.csv')
# # add_cli_significance_to_file('../data/supplementary4.csv','../data/supplementary4_significance.csv')

# # %%
# summary_self_interactions(400)
# # %%
# compare_actual_ppi_3_databases(400,True)
# # %%

# generate_set_of_pairs(['P25788-P28070', 'Q12824-Q96GM5','P11-P11'])
# # %%
