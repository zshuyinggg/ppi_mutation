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

    cookies = {
        'ncbi_sid': '450E00A929E75573_0232SID',
        '_ga_HF818T9R4Y': 'GS1.1.1654551992.1.0.1654551992.0',
        '_gid': 'GA1.2.1076087552.1665160675',
        'QSI_SI_aVGfI9HAVEqy8FE_intercept': 'true',
        '_gaexp': 'GAX1.2.oNcmAZfrT2W8UDVrczVjPw.19364.1',
        'WebEnv': '1FiowbyqiMJqixg09x5YMdoAfvhoeMFLrbMkcvvzLQSkZ%40450E00A929E75573_0232SID',
        '_ga_DP2X732JSX': 'GS1.1.1665160675.3.1.1665161569.0.0.0',
        '_ga': 'GA1.1.1240771573.1654551992',
        'ncbi_pinger': 'N4IgDgTgpgbg+mAFgSwCYgFwgCIAYBC2ATABwDMAggKICcAjA7lQKy5u5kDC+VAYvjXz4A7CU4A6OuIC2cZnRAAaEAFcAdgBsA9gENUaqAA8ALplBFMIAM5qwF5WUs2wSkABZL0AGbO40MFoQpsrMlq5EuJZ4hKSUtAx0TKzsXDz8giJikjJyCspECljORBjOGN6+/oHGGAByAPK1VOEWWADuHeJqAMYARshdGtJdyIjiAOZaMOE0lnRubpEOkVg0bK5khSDzixutICREjg6OWMYQKlAbHmcXVw4kYQ6zWHQAbMLCb5uuu69vJFwNCIb1+pxAuHEZGE4iW7huqk0un0RmC7lCWDhzHBJBIoRCCOEZA8IVBWGEWOEljhby2RBoH1cbzJIGg52QsCuAF8uUA==',
    }

    headers = {
        'authority': 'www.ncbi.nlm.nih.gov',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
        'cache-control': 'max-age=0',
        # Requests sorts cookies= alphabetically
        # 'cookie': 'ncbi_sid=450E00A929E75573_0232SID; _ga_HF818T9R4Y=GS1.1.1654551992.1.0.1654551992.0; _gid=GA1.2.1076087552.1665160675; QSI_SI_aVGfI9HAVEqy8FE_intercept=true; _gaexp=GAX1.2.oNcmAZfrT2W8UDVrczVjPw.19364.1; WebEnv=1FiowbyqiMJqixg09x5YMdoAfvhoeMFLrbMkcvvzLQSkZ%40450E00A929E75573_0232SID; _ga_DP2X732JSX=GS1.1.1665160675.3.1.1665161569.0.0.0; _ga=GA1.1.1240771573.1654551992; ncbi_pinger=N4IgDgTgpgbg+mAFgSwCYgFwgCIAYBC2ATABwDMAggKICcAjA7lQKy5u5kDC+VAYvjXz4A7CU4A6OuIC2cZnRAAaEAFcAdgBsA9gENUaqAA8ALplBFMIAM5qwF5WUs2wSkABZL0AGbO40MFoQpsrMlq5EuJZ4hKSUtAx0TKzsXDz8giJikjJyCspECljORBjOGN6+/oHGGAByAPK1VOEWWADuHeJqAMYARshdGtJdyIjiAOZaMOE0lnRubpEOkVg0bK5khSDzixutICREjg6OWMYQKlAbHmcXVw4kYQ6zWHQAbMLCb5uuu69vJFwNCIb1+pxAuHEZGE4iW7huqk0un0RmC7lCWDhzHBJBIoRCCOEZA8IVBWGEWOEljhby2RBoH1cbzJIGg52QsCuAF8uUA==',
        'referer': 'https://www.ncbi.nlm.nih.gov/clinvar/?gr=0&term=rs3829740',
        'sec-ch-ua': '"Google Chrome";v="105", "Not)A;Brand";v="8", "Chromium";v="105"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'same-origin',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36',
    }

    response = requests.get('https://www.ncbi.nlm.nih.gov/snp/rs3829740/', cookies=cookies, headers=headers).text
    # page=requests.get('https://www.ncbi.nlm.nih.gov/snp/'+str(snpid)).text
    soup=BS(response,'html.parser')

    print(soup.prettify())


get_clinvar_page('rs3829740')

# map_stringdb_uniprot()