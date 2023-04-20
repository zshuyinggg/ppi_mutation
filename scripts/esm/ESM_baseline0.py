import torch
from torch.utils.data import Dataset

global top_path  # the path of the top_level directory
global script_path, data_path, logging_path
import os, sys
from torch.utils.data import DataLoader
import esm

def find_current_path():
    if getattr(sys, 'frozen', False):
        # The application is frozen
        current = sys.executable
    else:
        # The application is not frozen
        # Change this bit to match where you store your data files:
        current = __file__

    return current


top_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(find_current_path()))))
sys.path.append(top_path)
script_path, data_path, logging_path= os.path.join(top_path,'scripts'),\
    os.path.join(top_path,'data'),\
    os.path.join(top_path,'logs')
from scripts.utils import *
from scripts.esm.model import *

from scripts.esm.datasets import *
import pandas as pd
if __name__ == '__main__':
    dataset = ProteinSequence(os.path.join(script_path, 'merged_2019_1.csv'),
                           data_path + '/2019_1_sequences_terminated.csv', gen_file=False, all_uniprot_id_file=
                              os.path.join(data_path, 'single_protein_seq/uniprotids_humap_huri.txt'),
                              test_mode=False,
                              #transform=transforms.Compose([
                                  #RandomCrop(512),
                   #               ToTensor()]
                              )



dataloader = DataLoader(dataset, batch_size=2,
                        shuffle=True, num_workers=1)
esm_mlp=Esm_mlp(mlp_input_dim=320,mlp_hidden_dim=160)
trainer=pl.Trainer(max_epochs=10, accelerator="gpu")
trainer.fit(model=esm_mlp,train_dataloaders=dataloader)

#checkpoint
# checkpoint=os.path.join(logging_path,)



# %%
