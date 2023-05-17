import torch
from lightning.pytorch.callbacks import ModelCheckpoint
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
from scripts.myesm.model import *

from scripts.myesm.datasets import *
import pandas as pd

if __name__ == '__main__':
    proSeq=ProteinSequence(test_mode=False)
    train_set,_= split_train_val(proSeq,1)

    print('Splitting all data set by length\n=======================')
    train_short_set,train_medium_set,train_long_set=cut_seq(train_set,0,512,1024,3000,True)


    print('\n\n\n Defining dataloaders...\n\n\n')
    train_short_dataloader = DataLoader(train_short_set, batch_size=2,
                            shuffle=False, num_workers=20)

    train_medium_dataloader = DataLoader(train_medium_set, batch_size=1,
                                        shuffle=False, num_workers=20)

    train_long_dataloader = DataLoader(train_long_set, batch_size=1,
                                       shuffle=False, num_workers=20)


    esm=Esm_infer()

    print('Starting training for short seqences....')
    pred_train_short_writer=CustomWriter(output_dir=data_path, prefix='all_short',write_interval="epoch")
    trainer=pl.Trainer(accelerator="gpu", devices=2, num_nodes=3, strategy="ddp",  callbacks=[pred_train_short_writer])
    trainer.predict(esm,train_short_dataloader ,return_predictions=False)



    print('Starting training for medium seqences....')
    pred_train_medium_writer=CustomWriter(output_dir=data_path, prefix='all_medium',write_interval="epoch")
    trainer=pl.Trainer(accelerator="gpu", devices=2, num_nodes=3, strategy="ddp",  callbacks=[pred_train_medium_writer])
    trainer.predict(esm,train_medium_dataloader ,return_predictions=False)



    print('Starting training for long seqences....')
    pred_train_long_writer=CustomWriter(output_dir=data_path, prefix='all_long',write_interval="epoch")
    trainer=pl.Trainer(accelerator="gpu", devices=2, num_nodes=3, strategy="ddp",  callbacks=[pred_train_long_writer])
    trainer.predict(esm,train_long_dataloader ,return_predictions=False)


