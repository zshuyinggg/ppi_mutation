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



train_val_split=0.8
if __name__ == '__main__':
    proSeq=ProteinSequence()
    train_set,val_set= split_train_val(proSeq,0.9)

    print('Splitting training set by length\n=======================')
    train_short_set,train_medium_set,train_long_set=cut_seq(train_set,0,512,1024,2048,True)
    print('Splitting validation set by length\n=======================')
    val_short_set,val_medium_set,val_long_set=cut_seq(val_set,0,512,1024,2048,True)


    print('\n\n\n Defining dataloaders...\n\n\n')
    train_short_dataloader = DataLoader(train_short_set, batch_size=4,
                            shuffle=True, num_workers=20)
    val_short_dataloader = DataLoader(val_short_set, batch_size=4,
                                  shuffle=False, num_workers=20)
    train_medium_dataloader = DataLoader(train_medium_set, batch_size=2,
                                        shuffle=True, num_workers=20)
    val_medium_dataloader = DataLoader(val_medium_set, batch_size=2,
                                      shuffle=False, num_workers=20)
    train_long_dataloader = DataLoader(train_long_set, batch_size=1,
                                       shuffle=True, num_workers=20)
    val_long_dataloader = DataLoader(val_long_set, batch_size=1,
                                      shuffle=False, num_workers=20)

    myesm=Esm_finetune()
    logger=TensorBoardLogger(os.path.join(logging_path,'esm_finetune'),name="esm2_t12_35M_UR50D",version='lr4-05_unfreeze6')


    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=True, mode="min")
    # checkpoint_callback = ModelCheckpoint(
    #         monitor='val_loss',
    #     dirpath= logging_path,
    #     filename='esm2_t30_150M_UR50D_lr1-03_-{epoch:02d}-{val_loss:.2f}'
    # )

    # trainer=pl.Trainer(max_epochs=80, logger=logger,devices=2, num_nodes=1, strategy=DDPStrategy(find_unused_parameters=True), accelerator="gpu",default_root_dir=logging_path, callbacks=[early_stop_callback])
    trainer=pl.Trainer(max_epochs=80, logger=logger,accelerator="gpu",default_root_dir=logging_path, callbacks=[early_stop_callback])
    print('=========Start to train short sequences============\n------------------------')
    trainer.fit(model=myesm,train_dataloaders=train_short_dataloader,val_dataloaders=val_short_dataloader)

    print('=========Start to train medium sequences============\n------------------------')
    trainer.fit(model=myesm,train_dataloaders=train_medium_dataloader,val_dataloaders=val_medium_dataloader)
#TODO: edit the training loop to train them all together in each epoch

    # trainer=pl.Trainer(max_epochs=10, logger=logger, accelerator="gpu",default_root_dir=logging_path, callbacks=[early_stop_callback])
    print('=========Start to train long sequences============\n------------------------')
    trainer.fit(model=myesm,train_dataloaders=train_long_dataloader,val_dataloaders=val_long_dataloader)
