import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset
global top_path  # the path of the top_level directory
global script_path, data_path, logging_path
import os, sys
from torch.utils.data import DataLoader
import esm
from argparse import ArgumentParser
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

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

train_val_split=0.8
if __name__ == '__main__':
    seq_dataset = ProteinSequence(os.path.join(script_path, 'merged_2019_1.csv'),
                           data_path + '/2019_1_sequences_terminated.csv', gen_file=False, all_uniprot_id_file=
                              os.path.join(data_path, 'single_protein_seq/uniprotids_humap_huri.txt'),
                              test_mode=False,
                              #transform=transforms.Compose([
                                  #RandomCrop(512),
                   #               ToTensor()]
                              )
seq_dataset.sort()
train_set,valid_set= split_train_val(seq_dataset,train_val_split)
train_dataloader = DataLoader(train_set, batch_size=24,
                        shuffle=False, num_workers=10)
val_dataloader = DataLoader(valid_set, batch_size=24,
                              shuffle=False, num_workers=10)
esm_mlp=Esm_mlp(mlp_input_dim=320,mlp_hidden_dim=160)


early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="max")
checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
    dirpath= logging_path,
    filename='esm_mlp_320_160_bz24_lr1-03_-{epoch:02d}-{val_loss:.2f}'
)
trainer=pl.Trainer(max_epochs=10, accelerator="gpu",default_root_dir=logging_path, callbacks=[early_stop_callback])
trainer.fit(model=esm_mlp,train_dataloaders=train_dataloader,val_dataloaders=val_dataloader)

#checkpoint
# checkpoint=os.path.join(logging_path,)


#TODO sort the batch to better make use of infer_on_cpu
# %%
