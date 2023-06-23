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
parser.add_argument('--esm', type=str, help='',default="esm2_t36_3B_UR50D")
parser.add_argument('--includewild', default=True, action=argparse.BooleanOptionalAction)

args = parser.parse_args()

num_devices=args.numdevices
num_nodes=args.numnodes
unfreeze_layers=args.unfreeze
esm_model=args.esm



if __name__ == '__main__':
    print('if_wild is set to true' if args.includewild else 'if_wild is set to false',flush=True)
    seed_everything(42, workers=True)
    proData=ProteinDataModule(train_val_ratio=0.8,low=0,medium=512,high=1028,veryhigh=1500,discard=True,num_devices=num_devices,num_nodes=num_nodes,delta=True,bs_short= 2,bs_medium=2,bs_long=2)
    myesm=Esm_finetune_delta(esm_model=eval("esm.pretrained.%s()"%esm_model),esm_model_dim=2560,n_class=2,repr_layers=36,unfreeze_n_layers=unfreeze_layers,lr=12*1e-5,random_crop_len=512,include_wild=args.includewild,debug=False).load_from_checkpoint('/scratch/user/zshuying/ppi_mutation/logs/esm_finetune_delta_ddp/esm2_t36_3B_UR50D/trainval0.8_lr1-05_esm_finetune_delta_wild_unfreeze_6/checkpoints/epoch=5-step=2562.ckpt')
    # myesm=Esm_finetune_delta(esm_model=eval("esm.pretrained.%s()"%esm_model),esm_model_dim=2560,n_class=2,repr_layers=36,unfreeze_n_layers=unfreeze_layers,lr=4*1e-5,random_crop_len=10,include_wild=args.includewild,debug=False)
    myesm.lr=4*1e-5
    myesm.debug=False
    logger=TensorBoardLogger(os.path.join(logging_path,'esm_finetune_delta_ddp'),name="%s"%esm_model,version='trainval0.8_lr1-05_esm_finetune_delta_wild_unfreeze_%s'%unfreeze_layers)
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
        trainer=pl.Trainer(max_epochs=200, 
                        logger=logger,
                        #    limit_train_batches=691,limit_val_batches=74, 
                        accelerator="gpu",
                        default_root_dir=logging_path, 
                        callbacks=[early_stop_callback],
                        reload_dataloaders_every_n_epochs=2,
                        plugins=[SLURMEnvironment(auto_requeue=False)])
                        #    reload_dataloaders_every_n_epochs=1)
        
    
    proData.trainer=trainer

    trainer.fit(myesm,datamodule=proData,ckpt_path='/scratch/user/zshuying/ppi_mutation/logs/esm_finetune_delta_ddp/esm2_t36_3B_UR50D/trainval0.8_lr1-05_esm_finetune_delta_wild_unfreeze_6/checkpoints/epoch=5-step=2562.ckpt') #need to use this to reload
    # trainer.fit(myesm,datamodule=proData) #need to use this to reload

    # trainer.fit(model=myesm,train_dataloaders=proData.train_dataloader(),val_dataloaders=proData.val_dataloader())
    #this does not reload because proData.train_dataloader( )returned a object and train_loaders just repeat call this object (not method)
    # trainer.fit(model=myesm,datamodule=proData)

