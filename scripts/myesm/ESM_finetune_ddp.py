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

from scripts.utils import *
from scripts.myesm.model import *
from scripts.myesm.datasets import *
import pandas as pd
parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('--test', type=int, help='1 if true; 0 if false',default=0)
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
    proData=ProteinDataModule(train_val_ratio=0.9,low=0,medium=512,high=1028,veryhigh=1500,discard=True,num_devices=num_devices,num_nodes=num_nodes,delta=False,bs_short=2,bs_medium=1)
    myesm=Esm_finetune(esm_model=eval("esm.pretrained.%s()"%esm_model) ,unfreeze_n_layers=unfreeze_layers,lr=10*1e-4)
    logger=TensorBoardLogger(os.path.join(logging_path,'esm_finetune_ddp'),name="%s"%esm_model,version='lr1-04_unfreeze%s_gather_investigate_10devices'%unfreeze_layers)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=20, verbose=True, mode="min")
    # checkpoint_callback = ModelCheckpoint(
    #         monitor='val_loss',
    #     dirpath= logging_path,
    #     filename='esm2_t36_3B_UR50D_-{epoch:02d}-{val_loss:.2f}'
    # )
    print('num devices %s, num node %s'%(num_devices,num_nodes))
    trainer=pl.Trainer(max_epochs=200, 
                       logger=logger,devices=num_devices, 
                       num_nodes=num_nodes, 
                       # limit_train_batches=691,limit_val_batches=74,
                       strategy=DDPStrategy(find_unused_parameters=True), 
                       accelerator="gpu",   
                       default_root_dir=logging_path, 
                       callbacks=[early_stop_callback],
                       plugins=[SLURMEnvironment(auto_requeue=False)],reload_dataloaders_every_n_epochs=1)


    # trainer=pl.Trainer(max_epochs=80, 
    #                    logger=logger,
    #                    # limit_train_batches=691,limit_val_batches=74,
    #                    accelerator="gpu",
    #                    default_root_dir=logging_path, 
    #                    callbacks=[early_stop_callback],
    #                    plugins=[SLURMEnvironment(auto_requeue=False)],reload_dataloaders_every_n_epochs=1)
    proData.trainer=trainer
    trainer.fit(model=myesm,datamodule=proData)

