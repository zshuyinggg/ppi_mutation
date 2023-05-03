import torch
from torch.utils.data import Dataset
global top_path  # the path of the top_level directory
global script_path, data_path, logging_path
import os, sys
from torch.utils.data import DataLoader
from torch import optim, nn, utils, Tensor
import lightning.pytorch as pl
import esm
def find_current_path():
    if getattr(sys, 'frozen', False):
        # The application is frozen
        current = sys.executable
    else:
        # The application is not frozen
        # Change this bit to match where you store your data files:
        current = __file__

    return current
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import BasePredictionWriter

top_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(find_current_path()))))
sys.path.append(top_path)
from scripts.utils import *


class MLP(nn.Module):
    def __int__(self,input_dim,hidden_dim=512):
        super.__init__()
        self.layers=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1),
            nn.Sigmoid
        )

    def forward(self,x):
        return self.layers(x)


class Esm_infer(pl.LightningModule):
    def __init__(self,esm_model=esm.pretrained.esm2_t6_8M_UR50D(),truncation_len=None):
        super().__init__()
        self.esm_model, alphabet=esm_model
        self.batch_converter=alphabet.get_batch_converter()
        self.alphabet=alphabet
        self.batch_dic={}
    def predict_step(self, batch, batch_idx):
        self.esm_model.eval()
        torch.cuda.empty_cache()

        idxes,labels,seqs=batch['idx'],batch['label'],batch['seq']
        self.batch_dic[batch_idx]=labels.to('cpu')
        batch_sample=list(zip(labels,seqs))
        del batch
        batch_labels, batch_strs, batch_tokens=self.batch_converter(batch_sample)
        batch_labels=torch.stack(batch_labels)
        batch_tokens=batch_tokens.to(self.device)
        batch_labels=batch_labels.to(self.device)
        batch_labels=(batch_labels+1)/2 #so ugly...

        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[6], return_contacts=False)
            token_representations = results["representations"][6]
            sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))
        del token_representations
        sequence_representations=torch.stack(sequence_representations)
        torch.cuda.empty_cache()
        return sequence_representations

class CustomWriter(BasePredictionWriter):

    def __init__(self, output_dir, prefix,write_interval="epoch"):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.prefix=prefix

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        torch.save(torch.vstack(predictions), os.path.join(self.output_dir, f"{self.prefix}_predictions_{trainer.global_rank}.pt"))

        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        # torch.save(batch_indices, os.path.join(self.output_dir, f"{self.prefix}_batch_indices_{trainer.global_rank}.pt"))
        current_ds=trainer.predict_dataloaders.dataset.dataset
        ds_indices=trainer.predict_dataloaders.dataset.indices[np.hstack(batch_indices[0])]
        vf=np.vectorize(lambda x:current_ds[x]['label'])
        labels=vf(ds_indices)
        torch.save(labels,os.path.join(self.output_dir, f"{self.prefix}_labels_{trainer.global_rank}.pt"))
    # def write_on_batch_end(self, trainer, prediction, batch, batch_idx, dataloader_idx #TODO: WHAT IS THIS
    # ):
    #     torch.save({"prediction":prediction,"label":batch['label']}, os.path.join(self.output_dir, dataloader_idx, f"{batch_idx}.pt"))
        


class Esm_mlp(pl.LightningModule):
    def __init__(self, mlp_input_dim, mlp_hidden_dim, esm_model=esm.pretrained.esm2_t6_8M_UR50D(),truncation_len=None,mixed_cpu=True ):
        super().__init__()
        self.save_hyperparameters()
        print(self.device)
        self.esm_model, alphabet=esm_model
        # self.esm_model.to(self.device)
        # self.register_buffer("alphabet", alphabet)

        self.mixed_cpu=mixed_cpu
        for param in self.esm_model.parameters():
            param.requires_grad = False
        self.batch_converter=alphabet.get_batch_converter()
        self.alphabet=alphabet
        #TODO: QUESTION: Do I need a relu or normalization after esm output?
        self.mlp=nn.Sequential(
            nn.Linear(mlp_input_dim,mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim,1),
            nn.Sigmoid()
        )


    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()

        self.mlp.train()
        #training loop definition
        idxes,labels,seqs=batch['idx'],batch['label'],batch['seq']
        batch_sample=list(zip(labels,seqs))
        del batch

        batch_labels, batch_strs, batch_tokens=self.batch_converter(batch_sample)
        batch_tokens=batch_tokens.to(self.device)
        batch_labels=torch.stack(batch_labels)
        batch_labels=batch_labels.to(self.device)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        batch_labels=(batch_labels+1)/2 #so ugly...
        # for sequences that are longer than 512, infer them on cpu
        # if self.mixed_cpu:
        #     self.log('mixed_cpu mode is on. Batch_len > 512 will be trained on cpu')
        #     if True in (batch_lens>512) :
        #         sequence_representations=self.infer_on_cpu(batch_tokens)
        #
        #     else:
        #         sequence_representations=self.infer_on_gpu(batch_tokens)
        #     self.mlp.cuda()
        sequence_representations=self.train_mul_gpu(batch_tokens)

        y=self.mlp(sequence_representations.float().to(self.device))
        loss=nn.functional.mse_loss(y,batch_labels)
        self.log('train_loss',loss)
        torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()

        self.mlp.eval()
        idxes,labels,seqs=batch['idx'],batch['label'],batch['seq']
        batch_sample=list(zip(labels,seqs))
        del batch
        batch_labels, batch_strs, batch_tokens=self.batch_converter(batch_sample)
        batch_labels=torch.stack(batch_labels)
        batch_tokens=batch_tokens.to(self.device)
        batch_labels=batch_labels.to(self.device)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        batch_labels=(batch_labels+1)/2 #so ugly...
        # for sequences that are longer than 512, infer them on cpu
        # if True in (batch_lens>512):
        #     sequence_representations=self.infer_on_cpu(batch_tokens)
        #
        # else:
        #     sequence_representations=self.infer_on_gpu(batch_tokens)
        # self.mlp.cuda()
        sequence_representations=self.train_mul_gpu(batch_tokens)

        y=self.mlp(sequence_representations.float())
        loss=nn.functional.mse_loss(y,batch_labels)
        self.log('val_loss',loss,on_step=True,on_epoch=True,sync_dist=True)
        torch.cuda.empty_cache()
        return loss

    def test_step(self, batch, batch_idx):
        self.mlp.eval()
        idxes,labels,seqs=batch['idx'],batch['label'],batch['seq']
        batch_sample=list(zip(labels,seqs))
        del batch
        batch_labels, batch_strs, batch_tokens=self.batch_converter(batch_sample)
        batch_labels=torch.stack(batch_labels)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        batch_labels=(batch_labels+1)/2 #so ugly...
        # for sequences that are longer than 512, infer them on cpu
        # if True in (batch_lens>512):
        #     sequence_representations=self.infer_on_cpu(batch_tokens)
        #
        # else:
        #     sequence_representations=self.infer_on_gpu(batch_tokens)
        # self.mlp.cuda()
        sequence_representations=self.train_mul_gpu(batch_tokens)
        y=self.mlp(sequence_representations.float().cuda())
        loss=nn.functional.mse_loss(y,batch_labels)
        self.log('test_loss',loss,on_step=True,on_epoch=True,sync_dist=True)
        torch.cuda.empty_cache()
        return loss


    def infer_on_cpu(self,batch_tokens):
        self.esm_model.to('cpu')
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[6], return_contacts=False)
            token_representations = results["representations"][6]
            sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))
        sequence_representations=torch.stack(sequence_representations)
        return sequence_representations.cuda()

    def infer_on_gpu(self,batch_tokens):
        self.esm_model.to('cuda')
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        batch_tokens=batch_tokens.cuda()
        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[6], return_contacts=False)
            token_representations = results["representations"][6]
            sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))
        sequence_representations=torch.stack(sequence_representations)
        return sequence_representations


    def train_mul_gpu(self,batch_tokens):
        self.esm_model.to(device=self.device)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        # batch_tokens=batch_tokens.to(device=self.device)
        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[6], return_contacts=False)
            token_representations = results["representations"][6]
            sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))
        del token_representations
        sequence_representations=torch.stack(sequence_representations)
        return sequence_representations

    def configure_optimizers(self,lr=1e-3) :
        optimizer=optim.Adam(self.parameters(),lr=lr)
        return optimizer




