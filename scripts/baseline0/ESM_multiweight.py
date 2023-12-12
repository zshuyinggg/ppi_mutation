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
script_path, data_path, logging_path = os.path.join(top_path, 'scripts'), \
                                       os.path.join(top_path, 'data'), \
                                       os.path.join(top_path, 'logs')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
torch.set_float32_matmul_precision('medium')
# os.environ["CUDA_LAUNCH_BLOCKING"]="1"
from scripts.utils import *
from scripts.baseline0.model import *
from scripts.baseline0.datasets import *
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, help='', default="b0_multiscale0_n_wild.yaml")
args = parser.parse_args()
config = load_config(args.config_file)
print(config)
# ckpt = '/scratch/user/zshuying/ppi_mutation/logs/esm_finetune_delta_ddp/2019/esm2_t6_8M_UR50D/83001_1050_multiweight_grace1050_multiweight_8bins_-epoch=04-val_loss=0.42.ckpt'
pj = os.path.join
if __name__ == '__main__':
    seed_everything(config['seed'], workers=True)
    logger = TensorBoardLogger(os.path.join(logging_path, 'Nov28'), name="%s" % args.config_file)
    proData = ProteinDataModule(logger, **config['data_init'])
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(logging_path, 'Nov28'),
                                          save_top_k=1,
                                          monitor="val_loss",
                                          mode="min",
                                          filename='%s_seed1050_' % args.config_file + "-{epoch:02d}-{val_loss:.2f}")
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10, verbose=True, mode="min")
    #
    if config.get('test'):
        myesm = Esm_delta_multiscale_weight(
            esm_model=eval("esm.pretrained.%s()" % config['model_init']['esm_model_name']),
            lr=config['data_init']['batch_size'] * config['num_devices'] * config['num_nodes'] * 1e-6,
            **config['model_init']).load_from_checkpoint(config['ckpt'])
    else:
        myesm = Esm_delta_multiscale_weight(
            esm_model=eval("esm.pretrained.%s()" % config['model_init']['esm_model_name']),
            lr=config['data_init']['batch_size'] * config['num_devices'] * config['num_nodes'] * 1e-6,
            **config['model_init'])

    if config['num_devices'] > 1:
        trainer = pl.Trainer(max_epochs=200,
                             logger=logger, devices=config['num_devices'],
                             num_nodes=config['num_nodes'],
                             strategy=DDPStrategy(find_unused_parameters=True),
                             accelerator="gpu",
                             default_root_dir=logging_path,
                             callbacks=[early_stop_callback, checkpoint_callback],
                             plugins=[SLURMEnvironment(auto_requeue=False)], reload_dataloaders_every_n_epochs=1)

    else:
        print('gpu training')
        trainer = pl.Trainer(max_epochs=200,
                             logger=logger,
                             accelerator="gpu",
                             default_root_dir=logging_path,
                             callbacks=[early_stop_callback])

    if config.get('test'):
        test_data = ProteinDataModule(logger, **config['test_init'])

        trainer.datamodule = test_data
        trainer.test(model=myesm, datamodule=test_data, ckpt_path=config['ckpt'])

    else:
        trainer.datamodule = proData

        trainer.fit(model=myesm, datamodule=proData)
