import torch
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
        current = sys.executable
    else:
        current = __file__
    return current

top_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(find_current_path()))))
sys.path.append(top_path)
script_path, data_path, logging_path= os.path.join(top_path,'scripts'),\
    os.path.join(top_path,'data'),\
    os.path.join(top_path,'logs')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from scripts.utils import *
from scripts.baseline0.model import *
from scripts.baseline0.datasets import *
import pandas as pd


parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('--config', type=str, help='',default="")
args = parser.parse_args()
all_args = get_config_dic(args.config)

if __name__ == '__main__':

    seed_everything(42, workers=True)
    proData=ProteinDataModule(all_args)
    myesm=Esm_finetune_delta(all_args)
    logger=TensorBoardLogger(os.path.join(logging_path,'test'),name="%s"%esm_model,version='trainval0.8_lr1-05_esm_finetune_delta_short_only_no_wild_unfreeze_%s'%unfreeze_layers)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=20, verbose=True, mode="min")
  
    if num_devices>1:
        trainer=pl.Trainer(max_epochs=200, 
                        logger=logger,devices=num_devices, 
                        num_nodes=num_nodes, 
                        # limit_train_batches=691,limit_val_batches=74,
                        strategy=DDPStrategy(find_unused_parameters=True), 
                        accelerator="gpu",
                        default_root_dir=logging_path, 
                        callbacks=[early_stop_callback],
                        plugins=[SLURMEnvironment(auto_requeue=False)],reload_dataloaders_every_n_epochs=2)

    else:
        trainer=pl.Trainer(max_epochs=80, 
                        logger=logger,
                        #    limit_train_batches=691,limit_val_batches=74,
                        accelerator="gpu",
                        default_root_dir=logging_path, 
                        callbacks=[early_stop_callback],
                        plugins=[SLURMEnvironment(auto_requeue=False)])
                        #    reload_dataloaders_every_n_epochs=1)
        

    proData.trainer=trainer
    trainer.fit(myesm,datamodule=proData) #need to use this to reload

    # trainer.fit(model=baseline0,train_dataloaders=proData.train_dataloader(),val_dataloaders=proData.val_dataloader())
    #this does not reload because proData.train_dataloader( )returned a object and train_loaders just repeat call this object (not method)
    # trainer.fit(model=baseline0,datamodule=proData)

