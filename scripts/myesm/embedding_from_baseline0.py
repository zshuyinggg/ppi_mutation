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
parser.add_argument('--esm_dim', type=int,default=640, help='')
parser.add_argument('--esm_layers', type=int,default=6, help='')
parser.add_argument('--esm', type=str, help='',default="esm2_t30_150M_UR50D")
parser.add_argument('--which_embds', type=str, default='0134',help="0:delta,1:wild,2:variant,3:local,4:AA")
parser.add_argument('--seed', type=int, default=42,help="random seed")
parser.add_argument('--version', type=str, default='debug',help="debug")

args = parser.parse_args()

num_devices=args.numdevices
num_nodes=args.numnodes
unfreeze_layers=args.unfreeze
esm_model=args.esm
ckpt='/scratch/user/zshuying/ppi_mutation/logs/esm_finetune_delta_ddp/2019/esm2_t6_8M_UR50D/83001_1050_multiweight_grace1050_multiweight_8bins_-epoch=04-val_loss=0.42.ckpt'
pj=os.path.join
if __name__ == '__main__':
    seed_everything(args.seed, workers=True)
    bs=20
    proData=AllProteinVariantData(clinvar_csv=pj(data_path,'clinvar','mutant_seq_2019_1_no_error.csv'),batch_size=20,num_workers=15)


    test_data=ProteinDataModule(test=True,clinvar_csv=pj(data_path,'clinvar/mutant_seq_2019_test_no_error.csv'),crop_val=True,train_val_ratio=0.000001,low=None,medium=None,high=None,veryhigh=None,discard=False,num_devices=num_devices,num_nodes=num_nodes,delta=True,bs_short= 2,bs_medium=2,bs_long=2,mix_val=True,train_mix=True,random_seed=args.seed)
    myesm=Esm_delta_multiscale_weight(save_embeddings=True,num_bins=8,bin_one_side_distance=[0,2,4,8,16,32,128,256],esm_model=eval("esm.pretrained.%s()"%esm_model),esm_model_dim=args.esm_dim,repr_layers=args.esm_layers,unfreeze_n_layers=unfreeze_layers,lr=bs*num_devices*num_nodes*1e-6).load_from_checkpoint(ckpt)
    myesm.save_embeddings=True
    if num_devices>1:
        trainer=pl.Trainer(max_epochs=200, 
                        devices=num_devices, 
                        num_nodes=num_nodes, 
                        strategy=DDPStrategy(find_unused_parameters=True), 
                        accelerator="gpu",
                        default_root_dir=logging_path, 
                        plugins=[SLURMEnvironment(auto_requeue=False)],reload_dataloaders_every_n_epochs=1)

    else:
        trainer=pl.Trainer(max_epochs=200, 
                        
                        accelerator="gpu",
                        default_root_dir=logging_path, 
                        reload_dataloaders_every_n_epochs=2,
                        plugins=[SLURMEnvironment(auto_requeue=False)])
        
    trainer.datamodule=proData
    # trainer.datamodule=test_data
    # trainer.test(model=myesm,datamodule=test_data,ckpt_path=ckpt)
    trainer.validate(model=myesm,datamodule=proData,ckpt_path=ckpt)

