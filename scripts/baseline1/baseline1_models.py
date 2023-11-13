from typing import Any, Optional
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
from lightning.pytorch.callbacks import BaseFinetuning
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
from torch_geometric.nn import GATConv, GINConv, GCNConv,SAGEConv

class MLP(nn.Module):
    def __init__(self, layer_num, hidden_dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.in_features=hidden_dim
        self.linear_list = nn.ModuleList( [nn.Linear(hidden_dim, hidden_dim) for _ in range(layer_num)] )
    def forward(self, x):
        for n, linear in enumerate(self.linear_list):
            x = linear(x)
            if (n + 1) == len(self.linear_list): break
            x = self.relu(x)
        return x



class plClassificationBaseModel(pl.LightningModule):
    def __init__(self, input_dim,hidden_dims,out_dim,dropout=False,lr=1e-4,*args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.input_dim,self.hidden_dims,self.out_dim,self.lr=input_dim,hidden_dims,out_dim,lr
        l=[input_dim]+list(hidden_dims)+[out_dim]
        self.mlp=nn.ModuleList([nn.Linear(l[i],l[i+1]) for i in range(len(l)-1)])
        self.relu=nn.ReLU()
        self.softmax=nn.Softmax(dim=1)
        self.ce_loss=nn.CrossEntropyLoss()
        self.train_out,self.val_out,self.test_out=[],[],[]
        self.auroc=BinaryAUROC()
        self.auprc=AveragePrecision(task='binary')
        if dropout:
            self.dropout=nn.Dropout(dropout) 
        else:self.dropout=False
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

    def on_test_epoch_end(self) :
        all_preds=torch.vstack(self.test_out)
        all_preds_gather=self.all_gather(all_preds).view(-1,3)
        test_auroc_gather=self.auroc(all_preds_gather.float()[:,1],all_preds_gather.long()[:,-1])
        test_auprc_gather=self.auprc(all_preds_gather.float()[:,1],all_preds_gather.long()[:,-1])
        test_loss=self.ce_loss(all_preds[:,:-1],all_preds.long()[:,-1])
        self.log('test_loss',test_loss,sync_dist=True)
        self.log('test_auroc_gathered',test_auroc_gather)
        self.log('test_auprc_gathered',test_auprc_gather)
        if self.trainer.global_rank==0:
            print('\n------gathered auroc is %s----\n'%test_auroc_gather,flush=True)
            print('\n------gathered auprc is %s----\n'%test_auprc_gather,flush=True)
        del all_preds, test_auroc_gather, test_auprc_gather,test_loss
        self.test_out.clear()

    def classify(self,x):
        for i,layer in enumerate(self.mlp):
            if self.dropout is not False:x=self.dropout(x)
            if i <len(self.mlp)-1:x=self.relu(layer(x))
            else:y=self.softmax(layer(x))
        return y


    def configure_optimizers(self,lr = None) :
        optimizer=optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),lr=lr if lr else self.lr)
        return optimizer
def define_act(f_act):
    if f_act=='relu':
        f = nn.ReLU()
        return f

    elif f_act=='tanh':
        f=nn.Tanh()
        return f
    else:return f_act #TODO 
class GNN(plClassificationBaseModel):
    def __init__(self, gnn_type, variant_initial_dim, variant_hidden_dim,wild_initial_dim, node_input_dim,num_gnn_layers,freeze=False,residual_strategy='mean',dim_reduction=False,dropout=0,f_act='relu', gat_attn_head=2,gin_mlp_layer=2, lr=1e-4, sampler=False,layers_dims=None,layer_norm=False,load_ckpt_before_training=None,**args):
        f_act=define_act(f_act)
        if residual_strategy == 'mean': dim2clf,layernorm1_dim=node_input_dim+variant_initial_dim, node_input_dim
        elif residual_strategy == 'stack' : dim2clf,layernorm1_dim = (num_gnn_layers+1)*node_input_dim+variant_initial_dim,node_input_dim*(num_gnn_layers+1)
        super().__init__(input_dim=dim2clf,hidden_dims=[],out_dim=2,dropout=dropout)
        self.node_dim,self.variant_initial_dim,self.wild_initial_dim=node_input_dim,variant_initial_dim,wild_initial_dim
        
        self.dim_reduction,self.residual_strategy,self.f=dim_reduction,residual_strategy,f_act
        if variant_initial_dim!=node_input_dim:
            if dim_reduction=='linear':
                self.variantDimReduction=nn.Sequential(nn.Dropout(dropout),nn.Linear(variant_initial_dim,node_input_dim))
            else:#TODO
                self.variantDimReduction=nn.Sequential(nn.Dropout(dropout),nn.Linear(variant_initial_dim,variant_hidden_dim),f_act,nn.Dropout(dropout),nn.Linear(variant_hidden_dim,node_input_dim))
        if wild_initial_dim!=node_input_dim:
            if dim_reduction=='linear':     
                self.wildDimReduction=nn.Sequential(nn.Dropout(dropout),nn.Linear(wild_initial_dim,node_input_dim))
            else:self.wildDimReduction=nn.Sequential(nn.Dropout(dropout),nn.Linear(wild_initial_dim,wild_initial_dim//2),f_act,nn.Dropout(dropout),nn.Linear(wild_initial_dim//2,node_input_dim))

        self.sampler=sampler
        if layers_dims: layers_dims=[node_input_dim] + layers_dims
        else: layers_dims=[node_input_dim]*(num_gnn_layers+1)
        if gnn_type == 'gat': 
            self.gnnconv_list = nn.ModuleList( [GATConv(in_channels=layers_dims[i], out_channels=layers_dims[i+1]//gat_attn_head, heads=gat_attn_head,dropout=dropout)
                                                                    for i in range(num_gnn_layers)] )
        elif gnn_type=='gcn':self.gnnconv_list=nn.ModuleList( [GCNConv(in_channels=node_input_dim,out_channels=node_input_dim)
                                                                    for _ in range(num_gnn_layers)] )
        elif gnn_type == 'gin': self.gnnconv_list = nn.ModuleList( [GINConv(MLP(gin_mlp_layer, node_input_dim),edge_dim=1)
                                                                    for _ in range(num_gnn_layers)] )
        
        self.node_input_dim=node_input_dim

        if layer_norm:
            self.layernorm=True
            self.layernorm1=nn.LayerNorm(layernorm1_dim)
            self.layernorm2=nn.LayerNorm(variant_initial_dim)
        else: self.layernorm=False
        if load_ckpt_before_training: 
            self.load_from_checkpoint(load_ckpt_before_training,**args)
            print('load from %s'%load_ckpt_before_training)
      
        self.save_hyperparameters()


    def training_step(self, batch, batch_idx):
        labels=batch.y[0]
        y=self.forward(batch)
        loss=self.ce_loss(y,labels)
        self.train_out.append(torch.hstack([y,labels.reshape(len(labels),1)]).cpu())
        return loss

    def validation_step(self, batch,batch_idx):
        labels=batch.y[0]
        y=self.forward(batch)
        loss=self.ce_loss(y,labels)
        self.val_out.append(torch.hstack([y,labels.reshape(len(labels),1)]).cpu())
        return loss

    def test_step(self, batch,batch_idx):
        labels=batch.y[0]
        y=self.forward(batch)
        loss=self.ce_loss(y,labels)
        self.test_out.append(torch.hstack([y,labels.reshape(len(labels),1)]).cpu())
        return loss
    

    def extract_merge_variant_from_layers(self,batch_size,x_reshaped,variant_indices):
        variants_aftergnn=[]
        for i in range(batch_size):
            if self.layernorm is not False:
                variants_aftergnn.append(self.layernorm1(x_reshaped[i,variant_indices[i],:].view(1,-1)))
            else:
                variants_aftergnn.append(x_reshaped[i,variant_indices[i],:].view(1,-1))
        return variants_aftergnn
    
    def get_gnn_embs(self,batch):
        node_embs,variant_embs,variant_idx=batch.x[0],batch.x[1],batch.x[2]
        edge_index=batch.edge_index[:batch.num_nodes,:batch.num_nodes]
        labels=batch.y[0]
        x=self.gnn(node_embs,edge_index)
        x_reshaped=x.view(len(labels),-1, self.node_dim)
        variants_aftergnn=x_reshaped[torch.arange(len(labels)),variant_idx]
        if self.layernorm is not False:variants_aftergnn=self.layernorm1(variants_aftergnn)
        return variants_aftergnn,labels

    def gnn(self, x, edge_index):
        # print(x.shape)
        x_stack=[x]
        x_sum = x
        for gnnconv in self.gnnconv_list:
            x = gnnconv(x=x, edge_index=edge_index)
            x = self.f(x)
            # print(x.shape)
            if self.residual_strategy=='mean':x_sum = x_sum + x 
            x_stack.append(x)
        if self.residual_strategy=='mean':x = x_sum / (len(self.gnnconv_list) + 1)
        elif self.residual_strategy=='stack':
            x=torch.hstack(x_stack)
            # print('stacking, x shape: %s'%str(x.shape))

        return x

    def merge_variant_embs_in_ppi(self, ppi_without_variant_embs,variant_embs,variant_indices):
        bs=len(variant_indices)
        all_embs_transformed=[]
        variant_embs=variant_embs.view(bs,self.variant_initial_dim)
        ppi_without_variant_embs=ppi_without_variant_embs.view(bs,-1, self.wild_initial_dim)
        if self.wild_initial_dim!=self.node_input_dim:
            all_wild_transfored=self.wildDimReduction(ppi_without_variant_embs)
        if self.variant_initial_dim!=self.node_input_dim:
            variant_transformed=self.variantDimReduction(variant_embs)
        for i in range(bs):
            wild_part1=all_wild_transfored[i,:variant_indices[i],:]
            wild_part2=all_wild_transfored[i,variant_indices[i]:,:]

            all_embs_transformed.append(torch.vstack([wild_part1,variant_transformed[i,:],wild_part2]))
        
        all_embs_transformed=torch.stack(all_embs_transformed,dim=0)
        return all_embs_transformed.view(-1,self.node_input_dim)

    def forward(self,batch):
        ppi_without_variant_embs,variant_embs,variant_indices,labels=batch.x[0],batch.x[1],batch.x[2],batch.y[0]
        ppi_embs_2_graph=self.merge_variant_embs_in_ppi(ppi_without_variant_embs,variant_embs,variant_indices)
        self.adj = SparseTensor.from_edge_index(batch.edge_index)
        num_nodes=ppi_embs_2_graph.shape[0]//len(labels)
        
        x=self.gnn(ppi_embs_2_graph,self.adj)
        x_reshaped=x.view(len(labels),num_nodes,-1) # (batch_size, num_node, )
        variants_aftergnn=torch.vstack(self.extract_merge_variant_from_layers(len(labels),x_reshaped,variant_indices))
        variants_beforegnn=variant_embs.view(len(labels),self.variant_initial_dim)
        if self.layernorm is not False:variants_beforegnn=self.layernorm2(variants_beforegnn)
        x2classify=torch.hstack([variants_beforegnn,variants_aftergnn])
        y=self.classify(x2classify)
        return y


class FreezeGNN(BaseFinetuning):
    def __init__(self) -> None:
        super().__init__()
    def freeze_before_training(self, pl_module) :
        self.freeze(pl_module.gnnconv_list)
        self.freeze([pl_module.variantDimReduction,pl_module.wildDimReduction,pl_module.layernorm1,pl_module.layernorm2])


class evaluate_graph_context(plClassificationBaseModel):
    def __init__(self, pretrained_gnn, input_dim,hidden_dims,out_dim,**args):
        super().__init__(input_dim,hidden_dims,out_dim)   
        self.pretrained_gnn=pretrained_gnn
        self.pretrained_gnn.freeze()  
        # self.save_hyperparameters() #question: dunno why this raises an error
    
    def training_step(self, batch,batch_idx):
        x,labels=self.pretrained_gnn.get_gnn_embs(batch)
        y=self.classify(x)
        loss=self.ce_loss(y,labels)
        self.train_out.append(torch.hstack([y,labels.reshape(len(labels),1)]).cpu())
        return loss
    def validation_step(self, batch,batch_idx):
        x,labels=self.pretrained_gnn.get_gnn_embs(batch)
        y=self.classify(x)
        loss=self.ce_loss(y,labels)
        self.val_out.append(torch.hstack([y,labels.reshape(len(labels),1)]).cpu())
        return loss
    def test_step(self, batch,batch_idx):
        x,labels=self.pretrained_gnn.get_gnn_embs(batch)
        y=self.classify(x)
        loss=self.ce_loss(y,labels)
        self.test_out.append(torch.hstack([y,labels.reshape(len(labels),1)]).cpu())
        return loss
    
        
class ESM_pretrained(pl.LightningModule):
    def __init__(self, esm_model,*args: Any, **kwargs: Any) :
        super().__init__(*args, **kwargs)
        self.val_embds={}
        self.esm_model, alphabet=esm_model
        self.batch_converter=alphabet.get_batch_converter()
        self.alphabet=alphabet

    def validation_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT | None:
        uniprots=batch['UniProt']
        batch_sample=[]
        seqs=[get_sequence_from_uniprot_id(uniprot) for uniprot in uniprots]
        for i,seq in enumerate(seqs):
            if len(seq)<1024:
                batch_sample.append((uniprots[i],seq))
            else:
                batch_sample.append((uniprots[i],seq[:1024]))
        wild_embs=self.get_esm_embedings(batch_sample)
        for i in range(wild_embs.shape[0]):
            self.val_embds[uniprots[i]]=wild_embs[i,:].cpu()
        
        return super().validation_step(*args, **kwargs)

    def on_validation_epoch_end(self) -> None:
        torch.save(self.val_embds,'/scratch/user/zshuying/ppi_mutation/data/baseline1/wild_esm_embds_%s.pt'%self.trainer.global_rank)

        return super().on_validation_epoch_end()

    def get_esm_embedings(self,batch_sample):
        _, _, batch_tokens=self.batch_converter(batch_sample)
        batch_tokens=batch_tokens.to(self.device)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        results = self.esm_model(batch_tokens, repr_layers=[6], return_contacts=False)
        token_representations = results["representations"][6]
        del results
        sequence_representations=[]
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))    
        sequence_representations=torch.vstack(sequence_representations)
        del batch_lens,batch_tokens
        return sequence_representations
        