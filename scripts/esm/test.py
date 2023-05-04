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
from lightning.pytorch.tuner import Tuner

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
script_path, data_path, logging_path= os.path.join(top_path,'scripts'), \
 os.path.join(top_path,'data'), \
 os.path.join(top_path,'logs')
from scripts.utils import *
from scripts.esm.model import *

from scripts.esm.datasets import *
import pandas as pd
if __name__ == '__main__':
  Embeddings=EsmMeanEmbeddings(if_initial_merge=False)
  train_set,val_set= split_train_val(Embeddings,0.8)
  mlp=myMLP(input_dim=320,hidden_dim=160)
  train_dataloader = DataLoader(train_set, batch_size=24,
                                      shuffle=True, num_workers=20)
  val_dataloader = DataLoader(val_set, batch_size=24,
                                    shuffle=False, num_workers=20)

  early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="max")
checkpoint_callback = ModelCheckpoint(
 monitor='val_auroc',
 dirpath= logging_path,
 filename='mlp_320_160_bz24_lr1-03_-{epoch:02d}-{val_loss:.2f}'
)



trainer=pl.Trainer(max_epochs=100, accelerator="gpu",default_root_dir=logging_path, callbacks=[early_stop_callback])
print('=========Start to train mlp============\n------------------------')
trainer.fit(model=mlp,train_dataloaders=train_dataloader,val_dataloaders=val_dataloader)


