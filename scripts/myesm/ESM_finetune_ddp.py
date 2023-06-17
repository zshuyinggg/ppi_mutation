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
parser.add_argument('--ini-file', default=None, required=True,
                    help='the path to the setting.ini')
args = parser.parse_args()

print(args.ini_file)
all_args = get_config_dic(args.ini_file)

if __name__ == '__main__':
    proData=ProteinDataModule(all_args)
    myesm=Esm_finetune(all_args)
    logger=TensorBoardLogger(os.path.join(logging_path,'esm_finetune_ddp'),name="%s"%all_args['esm_model'],version='lr1-04_unfreeze%s'%all_args['unfreeze_n_layers'])
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=int(all_args['patience']), verbose=True, mode="min")
    # checkpoint_callback = ModelCheckpoint(
    #         monitor='val_loss',
    #     dirpath= logging_path,
    #     filename='esm2_t36_3B_UR50D_-{epoch:02d}-{val_loss:.2f}'
    # )
    print('num devices %s, num node %s'%(all_args['num_devices'],all_args['num_nodes']))
    trainer=pl.Trainer(max_epochs=all_args['max_epochs'],
                       logger=logger,devices=all_args['num_devices'],
                       num_nodes=all_args['num_nodes'],
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

