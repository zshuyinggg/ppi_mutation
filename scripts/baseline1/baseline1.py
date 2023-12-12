#%%
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
global top_path  # the path of the top_level directory
global script_path, data_path, logging_path
import os, sys
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch import seed_everything
def find_current_path():
    if getattr(sys, 'frozen', False):current = sys.executable
    else:current = __file__
    return current
top_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(find_current_path()))))
sys.path.append(top_path)
script_path, data_path, logging_path= os.path.join(top_path,'scripts'),\
    os.path.join(top_path,'data'),\
    os.path.join(top_path,'logs')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
torch.set_float32_matmul_precision('medium')
from scripts.utils import *
from scripts.baseline0.model import *
from scripts.baseline1.datasets_baseline1 import *
from scripts.baseline1.baseline1_models import *
import argparse 

pj=os.path.join
parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, help='',default="gcn14_test.yaml")
args = parser.parse_args()



config = load_config(args.config_file)
print(config)
if __name__ == '__main__':
    seed_everything(1050, workers=True)

    variantPPI=VariantPPIModule(**config['data_init'])
    if config.get('test'):gcn=GNN(**config['gnn_init']).load_from_checkpoint(config['ckpt'],**config['gnn_init'])
    elif config.get('freeze'):gcn=GNN(**config['gnn_init']).load_from_checkpoint(config['ckpt_freeze'],**config['gnn_init'])
    else:gcn=GNN(**config['gnn_init'])
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10, verbose=True, mode="min")
    logger=TensorBoardLogger(os.path.join(logging_path,'Nov28','baseline1'),name="%s_seed1050"%args.config_file)
    checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="val_loss",
    mode="min",
    dirpath=os.path.join(logging_path,'Nov28','baseline1'),
    filename='%s_seed1050_'%args.config_file+"-{epoch:02d}-{val_loss:.2f}",
)
    if config['num_devices']>1:
            trainer=pl.Trainer(max_epochs=200, 
                            logger=logger,
                            devices=config['num_devices'], 
                            num_nodes=config['num_nodes'], 
                            strategy=DDPStrategy(), 
                            accelerator="gpu",
                            default_root_dir=logging_path, 
                            callbacks=[early_stop_callback,checkpoint_callback],
                            plugins=[SLURMEnvironment(auto_requeue=False)],reload_dataloaders_every_n_epochs=1)
    else:
        trainer=pl.Trainer(max_epochs=200, 
                        logger=logger,
                        accelerator="gpu",
                        default_root_dir=logging_path, 
                        reload_dataloaders_every_n_epochs=2,
                        plugins=[SLURMEnvironment(auto_requeue=False)])

    trainer.datamodule=variantPPI

    if not config.get('test'):trainer.fit(gcn,datamodule=variantPPI) 
    else:trainer.test(gcn,datamodule=variantPPI,ckpt_path=config['ckpt'])
    # else:trainer.validate(gcn,datamodule=variantPPI,ckpt_path=config['ckpt'])
    #%%
# %%
