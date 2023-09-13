import torch
import torch.cuda as cuda
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import Dataset
global top_path  # the path of the top_level directory
global script_path, data_path, logging_path
import os, sys
from torch.utils.data import DataLoader
import esm
import argparse 
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.plugins.environments import SLURMEnvironment
import os
from lightning.pytorch import seed_everything
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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
torch.set_float32_matmul_precision('medium')
# os.environ["CUDA_LAUNCH_BLOCKING"]="1"
from scripts.utils import *
from scripts.myesm.model import *
from scripts.myesm.datasets_baseline1 import *
import pandas as pd



pj=os.path.join

variantPPI=VariantPPIModule(root=pj(data_path,'baseline1'),clinvar_csv=pj(data_path,'clinvar','mutant_seq_2019_1_no_error.csv'),variant_embedding_path=pj(data_path,'baseline0','2019_variant_embds.pt'),
                            wild_embedding_path=pj(data_path,'baseline0','all_wild_embeddings.pt'),batch_size=20,num_workers=15,random_seed=1050,train_val_ratio=0.8)
variantPPI.dataset.process()
