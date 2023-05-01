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

proSeq=ProteinSequence()
train_set,val_set= split_train_val(proSeq,0.9)

print('Splitting training set by length\n=======================')
train_short_set,train_medium_set,train_long_set=cut_seq(train_set,0,512,1024,3000,True)
print('Splitting validation set by length\n=======================')
val_short_set,val_medium_set,val_long_set=cut_seq(val_set,0,512,1024,3000,True)


