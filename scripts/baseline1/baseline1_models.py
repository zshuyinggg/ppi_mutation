from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import numpy as np
import torch
from torch.utils.data import Dataset
global top_path  # the path of the top_level directory
global script_path, data_path, logging_path
# from transformers import AutoTokenizer, EsmForSequenceClassification
import os, sys
from torch.utils.data import DataLoader
from torch import optim, nn, utils, Tensor
import lightning.pytorch as pl
import esm
import torch.cuda as cuda
from datasets_baseline1 import *
import torch.nn.functional as F

from torchmetrics.classification import BinaryAUROC, MulticlassAUROC
from torchmetrics.classification import AveragePrecision
import re
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
from torch_geometric.nn import GATConv, GINEConv, GCNConv

class MLP(nn.Module):
    def __init__(self, layer_num, hidden_dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.linear_list = nn.ModuleList( [nn.Linear(hidden_dim, hidden_dim) for _ in range(layer_num)] )
    def forward(self, x):
        for n, linear in enumerate(self.linear_list):
            x = linear(x)
            if (n + 1) == len(self.linear_list): break
            x = self.relu(x)
        return x


class GNN(pl.LightningModule):
    def __init__(self, gnn_type, node_dim, num_gnn_layers, gat_attn_head=2,gin_mlp_layer=2, lr=1e-4, sampler=False):
        super().__init__()
        self.sampler=sampler
        if gnn_type == 'gat': self.gnnconv_list = nn.ModuleList( [GATConv(in_channels=node_dim, out_channels=node_dim//gat_attn_head, heads=gat_attn_head, edge_dim=1)
                                                                    for _ in range(num_gnn_layers)] )
        elif gnn_type=='gcn':self.gnnconv_list=nn.ModuleList( [GCNConv(in_channels=node_dim,out_channels=node_dim)
                                                                    for _ in range(num_gnn_layers)] )
        elif gnn_type == 'gin': self.gnnconv_list = nn.ModuleList( [GINEConv(nn.Sequential(MLP(gin_mlp_layer, node_dim)))
                                                                    for _ in range(num_gnn_layers)] )
        self.relu = nn.ReLU()
        self.node_dim=node_dim
        self.classifier=nn.Sequential(
            nn.Linear(node_dim*2,node_dim),
            nn.ReLU(),
            nn.Linear(node_dim,2),
            nn.Softmax(dim=1)
        )
        self.lr=lr
        self.train_out=[]
        self.val_out=[]
        self.ce_loss=nn.CrossEntropyLoss()
        self.auroc=BinaryAUROC()
        self.auprc=AveragePrecision(task='binary')

    def forward(self, x, edge_index):
        x_sum = x
        for gnnconv in self.gnnconv_list:
            x = self.relu(x)
            x = gnnconv(x=x, edge_index=edge_index)
            x_sum = x_sum + x
        x = x_sum / (len(self.gnnconv_list) + 1)
        return x

    def training_step(self, batch, batch_idx):
        if self.sampler:
            #get subgraph:
            random_walk_sampler=VariantRandomWalkSampler(batch,batch_size=4,walk_length=10)
            for subgraph in random_walk_sampler:
                pass
        else:
            node_embs,variant_embs,variant_indices=batch.x[0],batch.x[1],batch.x[2]
            self.adj = SparseTensor.from_edge_index(batch.edge_index)
            labels=batch.y[0]
            # print(labels)
            x=self.forward(node_embs,self.adj)
            x_reshaped=x.view(len(labels),-1, self.node_dim) # (batch_size, num_node, node_dim)
            # print(x_reshaped.shape)
            variants_aftergnn=[]
            for i in range(len(labels)):
                variants_aftergnn.append(x_reshaped[i,variant_indices[i],:].view(1,-1))
            variants_aftergnn=torch.vstack(variants_aftergnn)
            variants_beforegnn=variant_embs.view(len(labels),self.node_dim)
            x2classify=torch.hstack([variants_beforegnn,variants_aftergnn])
            y=self.classifier(x2classify)
            loss=self.ce_loss(y,labels)
            self.train_out.append(torch.hstack([y,labels.reshape(len(labels),1)]).cpu())
            print(torch.cuda.max_memory_reserved()/ 1024 ** 3 )
            return loss

    def configure_optimizers(self,lr = None) :
        optimizer=optim.Adam(self.parameters(),lr=lr if lr else self.lr)
        return optimizer
    
    def on_train_epoch_end(self) :
        all_preds=torch.vstack(self.train_out)
        all_preds_gather=self.all_gather(all_preds).view(-1,3)
        train_auroc_gather=self.auroc(all_preds_gather.float()[:,1],all_preds_gather.long()[:,-1])
        train_auprc_gather=self.auprc(all_preds_gather.float()[:,1],all_preds_gather.long()[:,-1])
        train_loss=self.ce_loss(all_preds[:,:-1],all_preds.long()[:,-1])
        self.log('train_loss',train_loss,sync_dist=True)
        self.log('train_auroc_gathered',train_auroc_gather)
        self.log('train_auprc_gathered',train_auprc_gather)
        if self.trainer.global_rank==0:
            print('\n------gathered auroc is %s----\n'%train_auroc_gather,flush=True)
            print('\n------gathered auprc is %s----\n'%train_auprc_gather,flush=True)
        del all_preds, train_auroc_gather, train_auprc_gather,train_loss
        self.train_out.clear()



    def validation_step(self, batch,batch_idx):
        node_embs,variant_embs,variant_idx=batch.x[0],batch.x[1],batch.x[2]
        edge_index=batch.edge_index[:batch.num_nodes,:batch.num_nodes]
        labels=batch.y[0]
        x=self.forward(node_embs,edge_index)
        x_reshaped=x.view(len(labels),-1, self.node_dim)
        variants_aftergnn=x_reshaped[torch.arange(len(labels)),variant_idx]
        variants_beforegnn=variant_embs.view(len(labels),self.node_dim)
        x2classify=torch.hstack([variants_beforegnn,variants_aftergnn])
        y=self.classifier(x2classify)
        loss=self.ce_loss(y,labels)
        self.val_out.append(torch.hstack([y,labels.reshape(len(labels),1)]).cpu())
        return loss


    def on_validation_epoch_end(self) :
        all_preds=torch.vstack(self.val_out)
        all_preds_gather=self.all_gather(all_preds).view(-1,3)
        val_auroc_gather=self.auroc(all_preds_gather.float()[:,1],all_preds_gather.long()[:,-1])
        val_auprc_gather=self.auprc(all_preds_gather.float()[:,1],all_preds_gather.long()[:,-1])
        val_loss=self.ce_loss(all_preds[:,:-1],all_preds.long()[:,-1])
        self.log('val_loss',val_loss,sync_dist=True)
        self.log('val_auroc_gathered',val_auroc_gather)
        self.log('val_auprc_gathered',val_auprc_gather)
        if self.trainer.global_rank==0:
            print('\n------gathered auroc is %s----\n'%val_auroc_gather,flush=True)
            print('\n------gathered auprc is %s----\n'%val_auprc_gather,flush=True)
        del all_preds, val_auroc_gather, val_auprc_gather,val_loss
        self.val_out.clear()