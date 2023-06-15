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

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
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


parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('--test', type=int, default=0,help='1 if true; 0 if false')
parser.add_argument('--numnodes', type=int, default=1,help='')
parser.add_argument('--numdevices', type=int,default=1, help='')
parser.add_argument('--unfreeze', type=int,default=10, help='')
parser.add_argument('--esm', type=str, help='',default="esm2_t36_3B_UR50D")
args = parser.parse_args()

num_devices=args.numdevices
num_nodes=args.numnodes
unfreeze_layers=args.unfreeze
esm_model=args.esm



if __name__ == '__main__':

    seed_everything(42, workers=True)
    proData=ProteinDataModule(train_val_ratio=0.8,low=0,medium=512,high=1028,veryhigh=1500,discard=True,num_devices=num_devices,num_nodes=num_nodes,delta=True,bs_short=2,bs_medium=1)
    myesm=Esm_finetune_delta(unfreeze_n_layers=unfreeze_layers,lr=12*1e-5)
    logger=TensorBoardLogger(os.path.join(logging_path,'esm_finetune_delta_ddp'),name="%s"%esm_model,version='trainval0.8_lr1-05_unfreeze_12devices_bs_short2_reload_every2epoch_%s'%unfreeze_layers)
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

    # trainer.fit(model=myesm,train_dataloaders=proData.train_dataloader(),val_dataloaders=proData.val_dataloader())
    #this does not reload because proData.train_dataloader( )returned a object and train_loaders just repeat call this object (not method)
    # trainer.fit(model=myesm,datamodule=proData)

