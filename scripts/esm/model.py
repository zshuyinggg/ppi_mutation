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

class Esm_mlp(pl.LightningModule):

    def __init__(self, mlp_input_dim, mlp_hidden_dim, esm_model=esm.pretrained.esm2_t6_8M_UR50D(),truncation_len=None ):
        super().__init__()
        self.esm_model, self.alphabet=esm_model
        for param in self.esm_model.parameters():
            param.requires_grad = False
        self.batch_converter=self.alphabet.get_batch_converter()
        #TODO: QUESTION: Do I need a relu or normalization after esm output?
        self.mlp=nn.Sequential(
            nn.Linear(mlp_input_dim,mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim,2),
            nn.Softmax()
        )

    def training_step(self, batch, batch_idx):
        #training loop definition
        idxes,labels,seqs=batch['idx'],batch['label'],batch['seq']
        batch_sample=list(zip(labels,seqs))
        del batch
        # TODO labels
        batch_labels, batch_strs, batch_tokens=self.batch_converter(batch_sample)
        batch_tokens=batch_tokens.cuda()
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[6], return_contacts=True)
        token_representations = results["representations"][6]
        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))
        sequence_representations=torch.transpose(torch.stack(sequence_representations),0,1)
        y=self.mlp(sequence_representations)
        loss=nn.functional.cross_entropy(y,batch_labels)
        self.log('train_loss',loss)
        torch.cuda.empty_cache()
        return loss

    def configure_optimizers(self,lr=1e-3) :
        optimizer=optim.Adam(self.parameters(),lr=lr)
        return optimizer