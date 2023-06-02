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
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.plugins.environments import SLURMEnvironment

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
from scripts.utils import *
from scripts.myesm.model import *
from scripts.myesm.datasets import *
import pandas as pd





if __name__ == '__main__':
    proData=ProteinDataModule(train_val_ratio=0.9,low=0,medium=512,high=1028,veryhigh=1500,discard=True)
    myesm=Esm_finetune(unfreeze_n_layers=8)
    logger=TensorBoardLogger(os.path.join(logging_path,'esm_finetune_ddp'),name="esm2_t12_35M_UR50D",version='lr4-05_unfreeze8')
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=True, mode="min")
    checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
        dirpath= logging_path,
        filename='esm2_t12_35M_UR50D_-{epoch:02d}-{val_loss:.2f}'
    )

    trainer=pl.Trainer(max_epochs=80, 
                       logger=logger,devices=2, 
                       num_nodes=2, 
                       # limit_train_batches=691,limit_val_batches=74,
                       strategy=DDPStrategy(find_unused_parameters=True), 
                       accelerator="gpu",
                       default_root_dir=logging_path, 
                       callbacks=[early_stop_callback],
                       plugins=[SLURMEnvironment(auto_requeue=False)],reload_dataloaders_every_n_epochs=1)


    trainer.fit(model=myesm,train_dataloaders=proData.train_dataloader(),val_dataloaders=proData.val_dataloader())

