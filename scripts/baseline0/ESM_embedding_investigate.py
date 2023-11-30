# %%
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

global top_path  # the path of the top_level directory
global script_path, data_path, logging_path
import os, sys
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch import seed_everything





def find_current_path():
    if getattr(sys, 'frozen', False):
        current = sys.executable
    else:
        current = __file__
    return current


top_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(find_current_path()))))
sys.path.append(top_path)
script_path, data_path, logging_path = os.path.join(top_path, 'scripts'), \
                                       os.path.join(top_path, 'data'), \
                                       os.path.join(top_path, 'logs')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
torch.set_float32_matmul_precision('medium')
from scripts.baseline0.model import *
from scripts.baseline0.datasets import *
import argparse

pj = os.path.join
parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, help='', default="b0_cls0_add_wild.yaml")
args = parser.parse_args()
config = load_config(args.config_file)
print(config)
if __name__ == '__main__':
    seed_everything(config['seed'], workers=True)
    bs = 20
    proData = ProteinDataModule(**config)
    logger = TensorBoardLogger(os.path.join(logging_path, 'b0_cls_token_training_with_wild'), name="%s" % args.config_file)

    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min",
                                          dirpath=os.path.join(logging_path, 'b0_cls_token_training_with_wild'),
                                          filename=args.config_file + "-{epoch:02d}-{val_loss:.2f}")
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10, verbose=True, mode="min")
    test_data = ProteinDataModule(clinvar_csv=pj(data_path, 'clinvar/mutant_seq_2019_test_no_error.csv'),
                                  train_val_ratio=0.000001, batch_size=config['batch_size'], random_seed=config['seed'])

    myesm = Esm_cls_token(**config)
    if config['num_devices'] > 1:
        trainer = pl.Trainer(max_epochs=200,
                             logger=logger,
                             devices=config['num_devices'],
                             num_nodes=config['num_nodes'],
                             strategy=DDPStrategy(find_unused_parameters=True),
                             accelerator="gpu",
                             default_root_dir=logging_path,
                             callbacks=[early_stop_callback, checkpoint_callback],
                             plugins=[SLURMEnvironment(auto_requeue=False)], reload_dataloaders_every_n_epochs=1)
    else:
        trainer = pl.Trainer(max_epochs=200,
                             accelerator="gpu",
                             default_root_dir=logging_path,
                             reload_dataloaders_every_n_epochs=2,
                             plugins=[SLURMEnvironment(auto_requeue=False)])

    trainer.datamodule = proData

    if not config.get('test'):
        trainer.fit(myesm, datamodule=proData)
    else:
        trainer.test(myesm, datamodule=test_data, ckpt_path=config['ckpt'])
