#%%
from Bio import SeqIO
import pandas as pd
records = list(SeqIO.parse("/scratch/user/zshuying/ppi_mutation/uniprots_huri_humap_uniref50.fasta", "fasta"))
cluster_df=pd.DataFrame(index=list(range(len(records))),columns=['name','cluster','cluster_size'])
for i,record in enumerate(records):
    description=record.description
    cluster=(' '.join(description.split(' ')[1:])).split('n=')[0]
    cluster_size=(' '.join(description.split(' ')[1:])).split('n=')[1].split(' ')[0]
    name=description.split(' ')[0]
    cluster_df.loc[i,'name'],cluster_df.loc[i,'cluster'],cluster_df.loc[i,'cluster_size']=name,cluster,str(cluster_size)

