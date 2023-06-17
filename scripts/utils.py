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

top_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(top_path)
script_path, data_path, logging_path= os.path.join(top_path,'scripts'), \
    os.path.join(top_path,'data'), \
    os.path.join(top_path,'logs')
def get_config_dic(config):
    """
    load in config dict from `configparser` type `ini` file
    """
    cf = configparser.ConfigParser()
    cf.read(config)
    return convert_config_type(cf)

def get_setting_name(string):
    base = os.path.basename(string)
    setting_name=os.path.splitext(base)[0]
    return setting_name
def convert_config_type(config):
    return {item[0]: eval(item[1]) for item in config.items('DEFAULT')}
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
    exclude'sp|Q6ZSR9|YJ005_HUMAN'
    """
    ids=set()
    for database in list_of_databases:
        print(database)
        if database=='huri':
            huri_set=set()

            huri_dir=os.path.join(data_path,'complexes/HuRI') #Ensembl pairs
            huri_uniprot_pairs=convert_huri_ensembl_2_uniprot([x[0].split('/')[-1] for x in os.walk(huri_dir)])
            l=[item.split('-') for item in huri_uniprot_pairs if '.' not in item and len(item.split('-'))==2]
            if '-' in l:l.remove('-')

            for item in l:
                if '|' in item[0] or '|' in item[1]: 
                    print('item %s found'%item)
                    continue
                else:
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
                if '|' in item[0] or '|' in item[1]: 
                    print('item %s found'%item)
                    continue
                else:
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


import functools

@functools.lru_cache(maxsize=128)
def get_sequence_from_uniprot_id_cached(id):
    url = f'https://www.uniprot.org/uniprot/{id}.fasta'
    response = requests.get(url)
    if response.ok:
        seq = ''.join(response.text.strip().split('\n')[1:])
        if seq.isupper():
            return seq
    return None

def get_sequence_from_uniprot_id(id):
    url='https://rest.uniprot.org/uniprotkb/%s.fasta'%id
    contnt = requests.get(url).text
    splt=contnt.split('\n')
    seq=''.join(splt[1:])
    if seq.isupper() and seq:
        return seq
    else: return None


def get_sequence_from_df(df):
    return [get_sequence_from_uniprot_id_cached(id) for id in df['UniProt']]



def gen_sequences_oneFile(list_of_ids,batch_size,out_file):
    total=len(list_of_ids)
    seq_dict={}
    for i,id in enumerate(tqdm(list_of_ids)):
        seq=get_sequence_from_uniprot_id(str(id))
        if seq:seq_dict[str(id)]=seq
        else:continue
    with open('%s.txt'%(out_file),'w') as f:
        f.writelines(str(seq_dict))
    return seq_dict


def gen_sequences_batch(list_of_ids,batch_size,out_file):
    """
    {'seq_id': 'protein_seq_1'

 'seq_primary': 'VQLVQSGAAVKKPGESLRISCKGSGYIFTNYWINWVRQMPGRGLEWMGRIDPSDSYTNYSSSFQGHVTISADKSISTVYLQWRSLKDTDTAMYYCARLGSTA'}
    """
    total=len(list_of_ids)
    n_f=math.ceil(total//batch_size)
    print(n_f)
    pbar1 = tqdm(total=n_f, position=1)
    for i in tqdm(range(n_f-1)):
        list_for_file=[]    
        temp_dic=dict()
        pbar2 = tqdm(total=batch_size, position=0)
        for id in tqdm(list_of_ids[i*batch_size:(i+1)*batch_size]):
            temp_dic=dict()
            temp_dic['seq_primary']=get_sequence_from_uniprot_id(str(id))
            temp_dic['seq_id']=str(id)
            if temp_dic['seq_primary']:list_for_file.append(temp_dic)
            pbar2.update(1)
            del temp_dic
        with open('%s_%d.json'%(out_file,i),'w') as f:
            json.dump(list_for_file,f)
            del list_for_file
        pbar1.update(1)

    list_for_file=[]    
    temp_dic=dict()  
    for id in list_of_ids[(n_f-1)*batch_size:]:
        temp_dic=dict()  
        temp_dic['seq_primary']=get_sequence_from_uniprot_id(str(id))
        temp_dic['seq_id']=str(id)
        if temp_dic['seq_primary']:list_for_file.append(temp_dic)
        del temp_dic
    pbar1.update(1)

    with open('%s_%d.json'%(out_file,n_f-1),'w') as f:
        json.dump(list_for_file,f)


def modify(seq,hgvs):
    #NP_000240.1:p.Ile219Val
    change=hgvs.split('p.')[1]
    obj=re.match(r'([a-zA-Z]+)([0-9]+)([a-zA-Z]+)',change)
    if obj is None:
        print('%s did not find match'%hgvs)
        new_seq='Error!! did not find match'
        return new_seq
    ori,pos,aft=obj.group(1),int(obj.group(2)),obj.group(3)
    ori=seq1(ori)
    if ori == '*': seq=seq+'*' # if the changes is made to the terminator, then the seq does not have it, we have to add it first
    aft=seq1(aft)  #TODO: '*' as a result should be removed!!
    try:assert ori==seq[pos-1]
    except (AssertionError,IndexError):
        print(hgvs)
        print ('ori=%s'%ori, seq[pos-1:pos+1])
        # raise AssertionError
        new_seq='Error!! Isoform is probably wrong!!!' #TODO: edit the code to deal with isoform

    new_seq=seq[:pos-1] + aft + seq[pos:]
    try:assert new_seq[pos-1]==aft

    except (AssertionError,IndexError):
        print(hgvs)
        print ('ori=%s'%ori, seq[pos-1:pos+1])
        # raise AssertionError
        new_seq='Error!! Isoform is probably wrong!!!' #TODO: edit the code to deal with isoform
    return new_seq


from dask.diagnostics import ProgressBar
ProgressBar().register()
@logger.catch
def gen_mutant_one_row(uniprot_id, name):
    print('getting sequence from uniprot_id %s'%uniprot_id)
    seq = get_sequence_from_uniprot_id(uniprot_id)
    print('get result as %s'%seq)
    if seq:
        print('modifing sequence')
        seq = modify(seq, name)
        print('sequence after modification is %s'%seq)
    else:
        print('result empty. seq marked as Error getting sequence from uniprot id')
        seq = 'Error getting sequence from uniprot id'

    return seq

def gen_mutant_from_df(df):
    return [gen_mutant_one_row(uniprot, name) for uniprot, name in zip(df['UniProt'], df['Name'])]


def gen_mutants_oneFile(list_of_ids,list_of_name,out_file,format='dict'):
    total=len(list_of_ids)
    seq_dict={}
    for i,id in enumerate(tqdm(list_of_ids)):
        seq=get_sequence_from_uniprot_id(str(id))
        if seq:seq_dict[str(id)]=modify(seq,list_of_name[i])
        else:continue
        
    with open('%s.txt'%(out_file),'w') as f:
        f.writelines(str(seq_dict))
    return seq_dict

def gen_mutants_batch(list_of_ids,list_of_name,batch_size,out_file):
    """
    {'seq_id': 'protein_seq_1'

 'seq_primary': 'VQLVQSGAAVKKPGESLRISCKGSGYIFTNYWINWVRQMPGRGLEWMGRIDPSDSYTNYSSSFQGHVTISADKSISTVYLQWRSLKDTDTAMYYCARLGSTA'}
    """
    total=len(list_of_ids)
    n_f=math.ceil(total//batch_size)
    pbar1 = tqdm(total=n_f, position=1)
    for i in tqdm(range(n_f-1)):
        list_for_file=[]    
        temp_dic=dict()
        pbar2 = tqdm(total=batch_size, position=0)
        for j,id in enumerate(tqdm(list_of_ids[i*batch_size:(i+1)*batch_size])):
            temp_dic=dict()
            seq=get_sequence_from_uniprot_id(str(id))
            if seq:
                temp_dic['seq_primary']=modify(seq,list_of_name[j])
            else: continue
            temp_dic['seq_id']=list_of_name[j]
            if temp_dic['seq_primary'].isupper():list_for_file.append(temp_dic)
            pbar2.update(1)
            del temp_dic
        with open('%s_%d.json'%(out_file,i),'w') as f:
            json.dump(list_for_file,f)
            del list_for_file
        pbar1.update(1)




    list_for_file=[]    
    temp_dic=dict()  
    for j,id in enumerate(list_of_ids[(n_f-1)*batch_size:]):
        temp_dic=dict()  
        seq=get_sequence_from_uniprot_id(str(id))
        if seq:
            temp_dic['seq_primary']=modify(seq,list_of_name[j])
        else: continue
        temp_dic['seq_id']=list_of_name[j]
        if temp_dic['seq_primary'].isupper():list_for_file.append(temp_dic)
        del temp_dic
    pbar1.update(1)

    with open('%s_%d.json'%(out_file,n_f-1),'w') as f:
        json.dump(list_for_file,f)
    

def check_sep_clinvar(ifuniprot,in_file,out_file):
    """
    I used ';' as sep to store the processed clinvar data from fullreleaseXML file, however, there is a column which has ';' itself. This function is to avoid mis-sep.
    """
    if 'sep' in in_file: 
        print('this file has already dealt with separation issue')
        return True
    if ifuniprot:cut_num=6
    else:cut_num=5
    with open(in_file,'r') as f:
        with open(out_file,'w') as f_out:
            line=f.readline()
            f_out.write('clinvar_id;review_status;clinical_sig;uniprot_kb;variant_type;hgvs_p;missense\n')
            while True:
                line=f.readline()

                if line.count(';')>cut_num:
                    discard_num=line.count(';')-cut_num+3
                    line_new=';'.join(line.split(';')[:3]+line.split(';')[discard_num:])
                    print(set(line.split(';'))-set(line_new.split(';')),' is discarded')
                    f_out.write(line_new)
                else:
                    f_out.write(line)
                if not line:break
    print('Processed this file to avoid separation issue with ;  and saved as %s'%out_file)


def get_uniprot(uniprot_kb_list):
    return [item.split('#')[0] for item in uniprot_kb_list]

def gen_mutants(df,out_file,bs=1280,delimiter=';'):
    uniprot_ls=get_uniprot(df['UniProt'])
    name_ls=df['Name'].tolist()
    if bs:
        seq_file=gen_mutants_batch(uniprot_ls,name_ls,bs,out_file)
    else:
        seq_file=gen_mutants_oneFile(uniprot_ls,name_ls,out_file)
        return seq_file

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
