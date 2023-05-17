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
from lightning.pytorch.callbacks import LearningRateFinder

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
from scripts.myesm.model import *
from lightning.pytorch.loggers import TensorBoardLogger
from scripts.myesm.datasets import *
import pandas as pd
if __name__ == '__main__':
  logger=TensorBoardLogger(os.path.join(logging_path,'esm_mlp_tb_logs'),name="esm_mlp_noval",version='640160hidden2bz')
  Embeddings=EsmMeanEmbeddings(if_initial_merge=False)
  train_set,val_set= split_train_val(Embeddings,0.8)
  mlp=myMLP(input_dim=320,hidden_dim1=640,hidden_dim2=160,learning_rate=1e-6)
  train_dataloader = DataLoader(train_set, batch_size=2,
                                      shuffle=True, num_workers=40)
  # val_dataloader = DataLoader(val_set, batch_size=24,
                                    # shuffle=False, num_workers=20)

  early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=0.001, patience=10, verbose=True, mode="min")
# checkpoint_callback = ModelCheckpoint(
#  monitor='val_auroc',
#  dirpath= data_path,
#  filename='mlp_320_160_bz24_lr1-06_-{epoch:02d}-{val_loss:.2f}'
# )

# lr_finder=LearningRateFinder()

trainer=pl.Trainer(max_epochs=100, logger=logger,accelerator="gpu",default_root_dir=logging_path, callbacks=[early_stop_callback])
print('=========Start to train mlp============\n------------------------')
# tuner = Tuner(trainer)

# finds learning rate automatically
# sets hparams.lr or hparams.learning_rate to that learning rate
# tuner.lr_find(mlp)


# trainer.fit(model=mlp,train_dataloaders=train_dataloader,val_dataloaders=val_dataloader)
trainer.fit(model=mlp,train_dataloaders=train_dataloader)


