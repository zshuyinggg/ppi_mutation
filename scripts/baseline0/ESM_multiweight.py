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
parser.add_argument('--config_file', type=str, help='', default="")
args = parser.parse_args()
config = load_config(args.config_file)
print(config)
# ckpt = '/scratch/user/zshuying/ppi_mutation/logs/esm_finetune_delta_ddp/2019/esm2_t6_8M_UR50D/83001_1050_multiweight_grace1050_multiweight_8bins_-epoch=04-val_loss=0.42.ckpt'
pj = os.path.join
if __name__ == '__main__':
    seed_everything(args.seed, workers=True)
    logger = TensorBoardLogger(os.path.join(logging_path, 'baseline0'), name="%s" % args.config_file)
    proData = ProteinDataModule(logger, **config['data_init'])
    checkpoint_callback = ModelCheckpoint(save_top_k=1,monitor="val_loss",mode="min",**config['callback_args'])
    early_stop_callback = EarlyStopping(**config['callback_args'])
    #
    # test_data = ProteinDataModule(test=True, clinvar_csv=pj(data_path, 'clinvar/mutant_seq_2019_test_no_error.csv'),
    #                               crop_val=True, train_val_ratio=0.000001, low=None, medium=None, high=None,
    #                               veryhigh=None, discard=False, num_devices=num_devices, num_nodes=num_nodes,
    #                               delta=True, bs_short=2, bs_medium=2, bs_long=2, mix_val=True, train_mix=True,
    #                               random_seed=args.seed)
    myesm = Esm_delta_multiscale_weight(esm_model=eval("esm.pretrained.%s()" % config['model_init']['esm_model_name']), lr= config['data_init']['batch_size'] * config['num_devices'] * config['num_nodes'] * 1e-6, **config['model_init'])
    print('so far all good')
    if config['num_devices'] > 1:
        trainer = pl.Trainer(max_epochs=200,
                             logger=logger, devices=num_devices,
                             num_nodes=num_nodes,
                             # limit_train_batches=691,limit_val_batches=74,
                             strategy=DDPStrategy(find_unused_parameters=True),
                             accelerator="gpu",
                             default_root_dir=logging_path,
                             callbacks=[early_stop_callback, checkpoint_callback],
                             plugins=[SLURMEnvironment(auto_requeue=False)], reload_dataloaders_every_n_epochs=1)

    else:
        trainer = pl.Trainer(max_epochs=200,
                             logger=logger,
                             #    limit_train_batches=691,limit_val_batches=74,
                             accelerator="gpu",
                             default_root_dir=logging_path,
                             callbacks=[early_stop_callback],
                             reload_dataloaders_every_n_epochs=2,
                             plugins=[SLURMEnvironment(auto_requeue=False)])
        #    reload_dataloaders_every_n_epochs=1)

    trainer.datamodule=proData
    # trainer.datamodule = test_data
    # trainer.test(model=myesm, datamodule=test_data, ckpt_path=ckpt)
    trainer.fit(model=myesm,datamodule=proData)
