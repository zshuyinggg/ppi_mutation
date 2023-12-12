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


# os.environ["CUDA_LAUNCH_BLOCKING"] = "1" #this will have problem in DDP setting! Use it only for single-device debugging!!
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
from scripts.baseline0.model import *
from scripts.baseline0.datasets import *
import pandas as pd


parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('--test', type=int, default=0,help='1 if true; 0 if false')
parser.add_argument('--numnodes', type=int, default=1,help='')
parser.add_argument('--numdevices', type=int,default=1, help='')
parser.add_argument('--unfreeze', type=int,default=6, help='')
parser.add_argument('--esm_dim', type=int,default=6, help='')
parser.add_argument('--esm_layers', type=int,default=6, help='')
parser.add_argument('--esm', type=str, help='',default="esm2_t30_150M_UR50D")
parser.add_argument('--which_embds', type=str, default='01',help="0:delta,1:wild,2:variant,3:local,4:AA,5:multiscale")
parser.add_argument('--seed', type=int, default=42,help="random seed")
parser.add_argument('--version', type=str, default=42,help="version number")
parser.add_argument('--ckpt', type=str, default=42,help="ckpt path")

args = parser.parse_args()

num_devices=args.numdevices
num_nodes=args.numnodes
unfreeze_layers=args.unfreeze
esm_model=args.esm

pj=os.path.join
ckpt=args.ckpt
if __name__ == '__main__':
    seed_everything(args.seed, workers=True)
    bs=20
    proData=AllProteinData(clinvar_csv=pj(data_path,'clinvar','mutant_seq_2019_1_no_error.csv'),batch_size=20,num_workers=20)

    test_data=ProteinDataModule(test=True,clinvar_csv=pj(data_path,'clinvar/mutant_seq_2019_test_no_error.csv'),crop_val=True,train_val_ratio=0.000001,low=None,medium=None,high=None,veryhigh=None,discard=False,num_devices=num_devices,num_nodes=num_nodes,delta=True,bs_short= 2,bs_medium=2,bs_long=2,mix_val=True,train_mix=True,random_seed=args.seed)
    myesm=Esm_finetune_delta(which_embds=args.which_embds).load_from_checkpoint(ckpt,which_embds=args.which_embds)


    trainer=pl.Trainer(max_epochs=200, 
                    logger=logger,devices=num_devices, 
                    num_nodes=num_nodes, 
                    strategy=DDPStrategy(find_unused_parameters=True), 
                    accelerator="gpu",
                    default_root_dir=logging_path, 
                    plugins=[SLURMEnvironment(auto_requeue=False)],reload_dataloaders_every_n_epochs=1)

    # trainer.callbacks[0].best_score=0
    trainer.validate(myesm,datamodule=proData,ckpt_path=ckpt) #need to use this to reload
    trainer.datamodule=test_data

    trainer.test(myesm,datamodule=test_data,ckpt_path=ckpt) #need to use this to reload

    # trainer.fit(model=baseline0,train_dataloaders=proData.train_dataloader(),val_dataloaders=proData.val_dataloader())
    #this does not reload because proData.train_dataloader( )returned a object and train_loaders just repeat call this object (not method)
    # trainer.fit(model=baseline0,datamodule=proData)
