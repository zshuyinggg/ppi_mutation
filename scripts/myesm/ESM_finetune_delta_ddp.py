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
parser.add_argument('--unfreeze', type=int,default=6, help='')
parser.add_argument('--esm_dim', type=int,default=6, help='')
parser.add_argument('--esm_layers', type=int,default=6, help='')
parser.add_argument('--esm', type=str, help='',default="esm2_t30_150M_UR50D")
parser.add_argument('--which_embds', type=str, default='01',help="0:delta,1:wild,2:variant,3:local,4:AA,5:bin")
parser.add_argument('--seed', type=int, default=42,help="random seed")
parser.add_argument('--version', type=str, default=42,help="version number")

args = parser.parse_args()

num_devices=args.numdevices
num_nodes=args.numnodes
unfreeze_layers=args.unfreeze
esm_model=args.esm

pj=os.path.join
# ckpt='/scratch/user/zshuying/ppi_mutation/logs/esm_finetune_delta_ddp/2022/esm2_t30_150M_UR50D/trainval0.8_lr1-05_esm_finetune_delta_wild_unfreeze_6_center_crop/checkpoints/epoch=32-step=5286.ckpt'
if __name__ == '__main__':
    seed_everything(args.seed, workers=True)
    bs=20
    proData=ProteinDataModule(clinvar_csv=pj(data_path,'clinvar','mutant_seq_2019_1_no_error.csv'),crop_val=True, train_val_ratio=0.8,low=None,medium=None,high=None,veryhigh=None,discard=False,num_devices=num_devices,num_nodes=num_nodes,delta=True,bs_short= bs,bs_medium=bs,bs_long=bs,mix_val=True,train_mix=True,random_seed=args.seed)
    myesm=Esm_finetune_delta(esm_model=eval("esm.pretrained.%s()"%esm_model),crop_val=True,esm_model_dim=args.esm_dim,crop_mode='center',n_class=2,repr_layers=args.esm_layers,unfreeze_n_layers=unfreeze_layers,lr=bs*num_devices*num_nodes*1e-6,crop_len=512,which_embds=args.which_embds,debug=False,balanced_loss=False,local_range=5)
    logger=TensorBoardLogger(os.path.join(logging_path,'esm_finetune_delta_ddp','2019'),name="%s"%esm_model,version='%s_seed_%d_trainval0.8_lr1-06_whichembds=%s_unfreeze_6_center_train_mix_cropval_bs_20'%(args.version,args.seed,args.which_embds))
    checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="val_loss",
    mode="min",
    dirpath=os.path.join(logging_path,'esm_finetune_delta_ddp','2019',esm_model),
    filename=str(args.version)+str(args.seed)+str(args.which_embds)+"-{epoch:02d}-{val_loss:.2f}",
)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10, verbose=True, mode="min")
    if num_devices>1:
        trainer=pl.Trainer(max_epochs=200, 
                        logger=logger,devices=num_devices, 
                        num_nodes=num_nodes, 
                        # limit_train_batches=691,limit_val_batches=74,
                        strategy=DDPStrategy(find_unused_parameters=True), 
                        accelerator="gpu",
                        default_root_dir=logging_path, 
                        callbacks=[early_stop_callback,checkpoint_callback],
                        plugins=[SLURMEnvironment(auto_requeue=False)],reload_dataloaders_every_n_epochs=1)

    else:
        trainer=pl.Trainer(max_epochs=200, 
                        logger=logger,
                        #    limit_train_batches=691,limit_val_batches=74, 
                        accelerator="gpu",
                        default_root_dir=logging_path, 
                        callbacks=[early_stop_callback],
                        reload_dataloaders_every_n_epochs=2,
                        plugins=[SLURMEnvironment(auto_requeue=False)])
                        #    reload_dataloaders_every_n_epochs=1)
        
    trainer.datamodule=proData
    trainer.fit(myesm,datamodule=proData) #need to use this to reload
    # trainer.validate(myesm,datamodule=proData) #need to use this to reload


