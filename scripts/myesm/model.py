from typing import Any
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
# from torch_geometric.nn import GATConv, GINEConv, GCNConv

import torch.nn.functional as F
import math

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
from lightning.pytorch.callbacks import BasePredictionWriter

top_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(find_current_path()))))
sys.path.append(top_path)
from scripts.utils import *
from scripts.myesm.datasets import ProteinSequence








class Esm_finetune(pl.LightningModule):
    def __init__(self, esm_model=esm.pretrained.esm2_t36_3B_UR50D(),esm_model_dim=2560,truncation_len=None,unfreeze_n_layers=10,repr_layers=36,lr=4*1e-3,crop_len=512):
        super().__init__()
        self.val_out=[]
        self.save_hyperparameters()
        self.esm_model_dim=esm_model_dim
        if isinstance(esm_model,str):self.esm_model, alphabet=eval(esm_model)
        else:self.esm_model, alphabet=esm_model
        self.batch_converter=alphabet.get_batch_converter()
        self.alphabet=alphabet
        self.proj=nn.Sequential(
            nn.Linear(esm_model_dim,2),
            nn.Softmax(dim=1)
        )
        self.train_out=[]
        self.unfreeze_n_layers=unfreeze_n_layers
        self.repr_layers=repr_layers
        self.freeze_layers()
        self.auroc=BinaryAUROC()
        self.lr=lr
        self.crop_len=crop_len


    def random_crop_batch(self,batch,batch_idx):
        names,seqs=batch['Name'],batch['seq']
        seqs_after=[]
        starts=[]
        positions=[]
        for i in range(len(batch['Name'])):
            name=names[i]
            seq=seqs[i]
            pos=self.get_pos_of_name(name)-1 #it counts from 1 in biology instead 0 in python
            if len(seq)>self.crop_len:
                np.random.seed(int('%d%d'%(self.trainer.current_epoch,batch_idx)))
                right=len(seq)-self.crop_len
                left=0
                min_start=max(left,pos-self.crop_len+1)
                max_start=min(right,pos)
                if pos>=len(seq):start=len(seq)-self.crop_len
                else: start=np.random.randint(low=min_start,high=max_start+1)
                seq_after=seq[start:start+self.crop_len]
            else:
                seq_after=seq
                start=None
            seqs_after.append(seq_after)
            starts.append(start)
            positions.append(pos)
        return seqs_after,starts,positions

    def center_crop_batch(self,batch,batch_idx):
        names,seqs=batch['Name'],batch['seq']
        seqs_after=[]
        starts=[]
        positions=[]
        for i in range(len(batch['Name'])):
            name=names[i]
            seq=seqs[i]
            pos=self.get_pos_of_name(name)-1 #it counts from 1 in biology instead 0 in python
            if len(seq)>self.crop_len:
                seq_after,start=self.center_crop(seq,pos)
            else:
                seq_after=seq
                start=None
            seqs_after.append(seq_after)
            starts.append(start)
            positions.append(pos)
        return seqs_after,starts,positions


    
    def get_pos_of_name(self,name):
        change=name.split('p.')[1]
        obj=re.match(r'([a-zA-Z]+)([0-9]+)([a-zA-Z]+)',change)
        if obj is None:
            print('%s did not find match'%name)
            new_seq='Error!! did not find match'
            return new_seq
        ori,pos,aft=obj.group(1),int(obj.group(2)),obj.group(3)
        return int(pos)

    def freeze_layers(self):
        num=self.repr_layers-self.unfreeze_n_layers
        for layer in self.esm_model.named_parameters():
            if 'layers' in layer[0] and int(layer[0].split('.')[1])<num:
                layer[1].requires_grad=False

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        labels,seqs=batch['label'].long(),batch['seq']
        if self.trainer.which_dl!='short':
            seqs,starts=self.crop_batch(batch,batch_idx)
        else:starts=None
        batch_sample=list(zip(labels,seqs))
        del batch,seqs,labels
        batch_labels, _, batch_tokens=self.batch_converter(batch_sample)
        # print(batch_tokens.shape)
        batch_size=batch_tokens.shape[0]

        batch_labels=torch.stack(batch_labels).reshape(batch_size)
        batch_labels=batch_labels.to(self.device)

        del batch_sample
        sequence_representations=self.train_mul_gpu(batch_tokens)
        y=self.proj(sequence_representations.float().to(self.device))
        
        loss=self.ce_loss(y,batch_labels)
        torch.cuda.empty_cache()
        self.train_out.append(torch.hstack([y,batch_labels.reshape(batch_size,1)]).cpu())
        del y,batch_labels,sequence_representations,batch_tokens
        return loss
    
    def on_train_epoch_end(self) :
        all_preds=torch.vstack(self.train_out)
        all_preds_gather=self.all_gather(all_preds).view(-1,3)
        train_auroc=self.auroc(all_preds.float()[:,1],all_preds.long()[:,-1])
        train_auroc_gather=self.auroc(all_preds_gather.float()[:,1],all_preds_gather.long()[:,-1])
        train_loss=self.ce_loss(all_preds[:,:-1],all_preds.long()[:,-1])
        self.log('train_loss',train_loss,sync_dist=True)
        self.log('train_auroc_gathered',train_auroc_gather)

        train_auroc_average=torch.mean(self.all_gather(train_auroc),dim=0)
        if self.trainer.global_rank==0:
            print('gathered auroc is %s'%train_auroc_gather)
        del all_preds, train_auroc,train_loss
        self.train_out.clear()

    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        self.esm_model.eval()
        self.proj.eval()
        labels,seqs=batch['label'].long(),batch['seq']
        batch_sample=list(zip(labels,seqs))
        del batch
        batch_labels, _, batch_tokens=self.batch_converter(batch_sample)
        batch_size=batch_tokens.shape[0]
        batch_labels=torch.stack(batch_labels).reshape(batch_size,1)
        batch_labels=batch_labels.to(self.device)
        sequence_representations=self.train_mul_gpu(batch_tokens)
        y=self.proj(sequence_representations.float().to(self.device))
        torch.cuda.empty_cache()
        pred=torch.hstack([y,batch_labels])
        self.val_out.append(pred.cpu())
        del labels,seqs,batch_sample,batch_tokens,batch_labels,sequence_representations,y
        return pred

    def on_validation_epoch_end(self):
        all_preds=torch.vstack(self.val_out)
        all_preds_gather=self.all_gather(all_preds).view(-1,3)
        val_auroc=self.auroc(all_preds.float()[:,1],all_preds.long()[:,-1])
        val_auroc_gather=self.auroc(all_preds_gather.float()[:,1],all_preds_gather.long()[:,-1])
        val_loss=self.ce_loss(all_preds[:,:-1],all_preds.long()[:,-1])
        self.log('val_auroc_gathered',val_auroc_gather)
        if self.trainer.global_rank==0:
            print('\n------gathered auroc is %s----\n'%val_auroc_gather)

        del all_preds, val_auroc,val_loss
        self.val_out.clear()


    def configure_optimizers(self,lr = None) :
        optimizer=optim.Adam(self.parameters(),lr=lr if lr else self.lr)
        return optimizer

    def train_mul_gpu(self,batch_tokens):
        torch.cuda.empty_cache()
        batch_tokens=batch_tokens.to(self.device)

        self.esm_model.to(device=self.device)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        # batch_tokens=batch_tokens.to(device=self.device)
        results = self.esm_model(batch_tokens, repr_layers=[self.repr_layers], return_contacts=False)
        token_representations = results["representations"][self.repr_layers]
        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))
        del token_representations
        torch.cuda.empty_cache()
        sequence_representations=torch.stack(sequence_representations)
        del results,batch_tokens
        return sequence_representations


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
        optimizer=optim.Adam(self.parameters(),lr=lr if lr else self.lr)
        return optimizer
    
class Esm_cls_token(plClassificationBaseModel):
    def __init__(self, esm_model=esm.pretrained.esm2_t36_3B_UR50D(),dropout=0.1,in_dim_clf=2560,repr_layers=36,lr=4*1e-3,crop_len=512,**args):
        super().__init__(input_dim=in_dim_clf,hidden_dims=[],out_dim=2,dropout=dropout,lr=lr)
        self.save_hyperparameters()
        if isinstance(esm_model,str):self.esm_model, alphabet=eval(esm_model)
        else:self.esm_model, alphabet=esm_model
        self.batch_converter=alphabet.get_batch_converter()
        self.alphabet,self.repr_layers,self.crop_len=alphabet,repr_layers,crop_len
        
    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        starts,labels,_,mutated_batch_samples,_=self.get_cropped_batch_samples(batch,batch_idx)
        mutated_embs=self.get_esm_embedings(mutated_batch_samples,starts=starts)
        y=self.classify(mutated_embs.float().to(self.device))
        del mutated_embs
        loss=self.ce_loss(y,labels)
        self.train_out.append(torch.hstack([y,labels.reshape(len(labels),1)]).cpu())
        del y,labels
        return loss
    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        starts,labels,_,mutated_batch_samples,_=self.get_cropped_batch_samples(batch,batch_idx)
        mutated_embs=self.get_esm_embedings(mutated_batch_samples,starts=starts)
        y=self.classify(mutated_embs.float().to(self.device))
        del mutated_embs
        loss=self.ce_loss(y,labels)
        self.val_out.append(torch.hstack([y,labels.reshape(len(labels),1)]).cpu())
        del y,labels
        return loss
    def test_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        starts,labels,_,mutated_batch_samples,_=self.get_cropped_batch_samples(batch,batch_idx)
        mutated_embs=self.get_esm_embedings(mutated_batch_samples,starts=starts)
        y=self.classify(mutated_embs.float().to(self.device))
        del mutated_embs
        loss=self.ce_loss(y,labels)
        self.test_out.append(torch.hstack([y,labels.reshape(len(labels),1)]).cpu())
        del y,labels
        return loss
    def get_pos_of_name(self,name):
        change=name.split('p.')[1]
        obj=re.match(r'([a-zA-Z]+)([0-9]+)([a-zA-Z]+)',change)
        if obj is None:
            print('%s did not find match'%name)
            new_seq='Error!! did not find match'
            return new_seq
        ori,pos,aft=obj.group(1),int(obj.group(2)),obj.group(3)
        return int(pos)
    def get_cropped_batch_samples(self,batch,batch_idx):
        labels=batch['label'].long()
        locs=batch['Loc'].long()
        seqs,starts,pos=self.crop_batch(batch,batch_idx)
        batch['seq']=seqs
        mutated_batch_samples=list(zip(locs,seqs))
        # wild_batch_samples=self.get_wild_batch(batch,starts=starts,pos=pos) 
        return starts,labels,locs,mutated_batch_samples
    def center_crop(self,seq,pos):
        left_half=self.crop_len//2
        right_half=self.crop_len-left_half
        ideal_right=pos+right_half
        ideal_left=pos-left_half

        actual_right=len(seq)
        actual_left=0
        if actual_left>ideal_left:left,right=actual_left,actual_left+self.crop_len
        else:
            if actual_right<ideal_right:left,right=actual_right-self.crop_len,actual_right
            else:left,right=ideal_left,ideal_right
        seq_after=seq[left:right]
        return seq_after,left
    def crop_batch(self,batch):
        names,seqs=batch['Name'],batch['seq']
        seqs_after=[]
        starts=[]
        positions=[]
        for i in range(len(batch['Name'])):
            name=names[i]
            seq=seqs[i]
            pos=self.get_pos_of_name(name)-1 #it counts from 1 in biology instead 0 in python
            if len(seq)>self.crop_len:
                seq_after,start=self.center_crop(seq,pos)
            else:
                seq_after=seq
                start=None
            seqs_after.append(seq_after)
            starts.append(start)
            positions.append(pos)
        return seqs_after,starts,positions

        
    def get_esm_embedings(self,batch_sample):
        torch.cuda.empty_cache()
        locs, _, batch_tokens=self.batch_converter(batch_sample)
        batch_tokens=batch_tokens.to(self.device)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        results = self.esm_model(batch_tokens, repr_layers=[self.repr_layers], return_contacts=False)
        token_representations = results["representations"][self.repr_layers]
        del results
        sequence_representations = []
        AA_representations=[]
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 0])    
            AA_representations.append(token_representations[i,locs[i]+1])
        del token_representations
        torch.cuda.empty_cache()
        sequence_representations=torch.vstack(sequence_representations)
        AA_representations=torch.vstack(AA_representations)
        del batch_lens,batch_tokens
        return torch.hstack([sequence_representations,AA_representations])

class Esm_Transformer(plClassificationBaseModel):
    def __init__(self, esm_model=esm.pretrained.esm2_t36_3B_UR50D(),dropout=0.1,nhead=4,esm_model_dim=2560,repr_layers=36,lr=4*1e-3,crop_len=512):
        super().__init__()
        self.save_hyperparameters()
        self.esm_model_dim=esm_model_dim
        if isinstance(esm_model,str):self.esm_model, alphabet=eval(esm_model)
        else:self.esm_model, alphabet=esm_model
        self.batch_converter=alphabet.get_batch_converter()
        self.alphabet=alphabet
        self.transformer=nn.Transformer(d_model=esm_model_dim, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=esm_model_dim*nhead, activation='tanh',layer_norm_eps=1e-05, batch_first=True, bias=True,dropout=dropout)
        self.esm_model.freeze()
    def get_cropped_batch_samples(self,batch,batch_idx):
        labels=batch['label'].long()
        locs=batch['Loc'].long()
        seqs,starts,pos=self.crop_batch(batch,batch_idx)
        batch['seq']=seqs
        mutated_batch_samples=list(zip(locs,seqs))
        wild_batch_samples=self.get_wild_batch(batch,starts=starts,pos=pos) 
        return starts,labels,locs,mutated_batch_samples,wild_batch_samples
    
    def get_wild_batch(self,mutated_batch,starts=None,pos=None):
        uniprots=mutated_batch['UniProt']
        mutants=mutated_batch['seq']
        locs=mutated_batch['Loc']
        mutant_lens=[len(seq) for seq in mutants]
        seqs=[get_sequence_from_uniprot_id(uniprot) for uniprot in uniprots]
        batch_sample=[]
        lens=[]
        for i,seq in enumerate(seqs):
            if starts[i]==0: #no random cropping
                # print('this seq is not cropped as the length is %s, starts[i] is %s'%(len(seq),starts[i]))
                lens.append(mutant_lens[i])
                batch_sample.append((locs[i],seq[:mutant_lens[i]])) #the wild seq has to be of same length as the mutant

            elif starts[i] is not None: #if mutated sequences are cropped and a start is returned
                seq=seq[starts[i]:starts[i]+self.crop_len]
                batch_sample.append((locs[i],seq))
                lens.append(len(seq))

            elif starts[i] is None and len(seq)>self.crop_len: # if mutated seq is not cropped but the wild is too long
                seq,_=self.crop(seq,pos[i])
                batch_sample.append((locs[i],seq))
                lens.append(len(seq))
            else:
                batch_sample.append((locs[i],seq))
                lens.append(len(seq))
        # print('\n length of wild batch is %s'%lens)
        return batch_sample
    

class Esm_finetune_delta(Esm_finetune):
    def __init__(self, esm_model=esm.pretrained.esm2_t12_35M_UR50D(),crop_val=True,esm_model_dim=480,n_class=2,truncation_len=None,unfreeze_n_layers=3,repr_layers=12,batch_sample='random',which_embds='01',lr=None,crop_len=512,debug=False,crop_mode='center',balanced_loss=False,local_range=0):
        """

        :param esm_model:
        :param esm_model_dim:
        :param n_class:
        :param truncation_len:
        :param unfreeze_n_layers:
        :param repr_layers:
        :param batch_sample: if "random", then each batch contains random mutated samples and will minus their corresponding wild embeddings.
        :param include_wild: if True, include wild embeddings as input for linear projections too.
        """
        super().__init__(esm_model,esm_model_dim,truncation_len,unfreeze_n_layers,repr_layers,lr,crop_len=crop_len)
        self.batch_sample=batch_sample
        self.which_embds=which_embds
        self.local_range=local_range
        self.init_dataset()
        self.auroc=BinaryAUROC()
        self.auprc=AveragePrecision(task='binary')
        self.embs_dim=self.get_embs_dim()
        self.proj=nn.Sequential(
            nn.Linear(self.embs_dim,2),
            nn.Softmax(dim=1)
        )
        self.val_out=[]
        self.test_out=[]
        self.train_out=[]
        self.debug=debug
        if crop_mode=='random':
            self.crop=self.random_crop
            self.crop_batch=self.random_crop_batch
        elif crop_mode=='center':
            self.crop=self.center_crop
            self.crop_batch=self.center_crop_batch
        self.crop_val=crop_val
        if balanced_loss:
            self.ce_loss=Loss(loss_type="cross_entropy",class_balanced=True,samples_per_class=[13902,7706]) 
            print('Using Balanced CE Loss')
        else:
            self.ce_loss=torch.nn.CrossEntropyLoss()
            print('Using Normal CE Loss')

    def get_embs_dim(self):
        doc=''
        dim=0
        # print(self.which_embds)
        # print(type(self.which_embds))
        if '0' in self.which_embds:
            dim+=self.esm_model_dim
            doc+'delta_mean_embs;'
            if '3' in self.which_embds:
                dim+=self.esm_model_dim
                doc+'delta_local_embs;'
            if '4' in self.which_embds:
                dim+=self.esm_model_dim
                doc+'delta_aa_embs;'
        if '1' in self.which_embds:
            dim+=self.esm_model_dim
            doc+'wild_mean_embs;'
            if '3' in self.which_embds:
                dim+=self.esm_model_dim
                doc+'wild_local_embs;'
            if '4' in self.which_embds:
                dim+=self.esm_model_dim
                doc+'wild_aa_embs;'
        if '2' in self.which_embds:
            dim+=self.esm_model_dim
            doc+'mutated_mean_embs;'
            if '3' in self.which_embds:
                dim+=self.esm_model_dim
                doc+'mutated_local_embs;'
            if '4' in self.which_embds:
                dim+=self.esm_model_dim
                doc+'mutated_aa_embs;'
        print('========',doc,'=========')
        return dim

    def init_dataset(self):
        self.dataset=ProteinSequence()

    def print_memory(self,step,detail=False):
        if self.debug: 
            mem = cuda.memory_allocated() / 1024 ** 3  
            print('\n====== %s memory used:%.2f=============\n'%(step,mem),flush=True)
            if detail and mem>10:
                for name, param in self.named_parameters():
                    if param.requires_grad:
                        param_memory = param.element_size() * param.numel() / 1024 ** 3  # Memory usage for the parameter
                        if param_memory>0.3:
                            print(f"{name} memory usage: {param_memory:.5f} GB",flush=True)

                for name, buffer in self.named_buffers():
                    buffer_memory = buffer.element_size() * buffer.numel() / 1024 ** 3  # Memory usage for the buffer
                    if buffer_memory>0.3:
                        print(f"{name} memory usage: {buffer_memory:.5f} GB",flush=True)

                for name, tensor in self.__dict__.items():
                    if torch.is_tensor(tensor) and tensor.is_cuda:
                        tensor_memory = tensor.element_size() * tensor.numel() / 1024 ** 3  # Memory usage for the tensor
                        if tensor_memory>0.3:print(f"{name} memory usage: {tensor_memory:.5f} GB",flush=True)

                for name, tensor in self.named_buffers():
                    if tensor.grad is not None:
                        tensor_grad_memory = tensor.grad.element_size() * tensor.grad.numel() / 1024 ** 3  # Memory usage for the tensor gradient
                        if tensor_grad_memory>0.3:print(f"{name}.grad memory usage: {tensor_grad_memory:.5f} GB",flush=True)

                for name, param in self.named_parameters():
                    if param.grad is not None:
                        param_grad_memory = param.grad.element_size() * param.grad.numel() / 1024 ** 3  # Memory usage for the parameter gradient
                        if param_grad_memory>0.3:print(f"{name}.grad memory usage: {param_grad_memory:.5f} GB",flush=True)
        
        else:
            pass
    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        self.print_memory('entering training step',detail=True)
        labels=batch['label'].long()
        # print(batch['Name'],batch['Loc'])
        locs=batch['Loc'].long()
        seqs,starts,pos=self.crop_batch(batch,batch_idx)
        batch['seq']=seqs
        # len_seqs=[len(seq) for seq in seqs]
        # print('===================================\nlength of seqs in this batch(%s) is %s\n========================='%(batch_idx,len_seqs),flush=True)
        mutated_batch_samples=list(zip(locs,seqs))
        del seqs
        wild_batch_samples=self.get_wild_batch(batch,starts=starts,pos=pos) 
        del batch
        self.print_memory('initiated batches',detail=True)
        mutated_embs=self.get_esm_embedings(mutated_batch_samples,loc=locs,starts=starts,local_range=self.local_range)
        self.print_memory('mutated_embs obtained',detail=True)
        wild_embs=self.get_esm_embedings(wild_batch_samples,starts=starts,loc=locs,local_range=self.local_range)
        self.print_memory('wild_embs obtained',detail=True)
        # print(mutated_embs,wild_embs)
        # print([len(sample[1]) for sample in wild_batch_samples])
        # return 0
            
        batch_size=len(labels)

        del mutated_batch_samples,wild_batch_samples

        embs=self.define_embds(wild_embs,mutated_embs,verbose=True)
        # print(embs)
        y=self.proj(embs.float().to(self.device))

        del mutated_embs,wild_embs,embs

        loss=self.ce_loss(y,labels)
        torch.cuda.empty_cache()
        self.train_out.append(torch.hstack([y,labels.reshape(batch_size,1)]).cpu())

        del y,labels

        return loss
    

    def define_embds(self,wild_embs,mutated_embs,verbose=False):

        l=[]
        doc=''
        wild_mean_embs,wild_local_embs,wild_AA_embs=wild_embs['sequence_representations'],wild_embs['local_representations'],wild_embs['AA_representations']
        mutated_mean_embs,mutated_local_embs,mutated_AA_embs=mutated_embs['sequence_representations'],mutated_embs['local_representations'],mutated_embs['AA_representations']
        delta_mean_embs=mutated_mean_embs-wild_mean_embs
        delta_local_embs=mutated_local_embs-wild_local_embs
        delta_AA_embs=mutated_AA_embs-wild_AA_embs

        if '0' in self.which_embds:
            l.append(delta_mean_embs)
            doc+'delta_mean_embs;'
            if '3' in self.which_embds:
                l.append(delta_local_embs)
                doc+'delta_local_embs;'
            if '4' in self.which_embds:
                l.append(delta_AA_embs)
                doc+'delta_aa_embs;'
        if '1' in self.which_embds:
            l.append(wild_mean_embs)
            doc+'wild_mean_embs;'
            if '3' in self.which_embds:
                l.append(wild_local_embs)
                doc+'wild_local_embs;'
            if '4' in self.which_embds:
                l.append(wild_AA_embs)
                doc+'wild_aa_embs;'
        if '2' in self.which_embds:
            l.append(mutated_mean_embs)
            doc+'mutated_mean_embs;'
            if '3' in self.which_embds:
                l.append(mutated_local_embs)
                doc+'mutated_local_embs;'
            if '4' in self.which_embds:
                l.append(mutated_AA_embs)
                doc+'mutated_aa_embs;'
        if verbose: print(doc)
        return torch.hstack(l)


    def on_train_epoch_end(self) :
        all_preds=torch.vstack(self.train_out)
        all_preds_gather=self.all_gather(all_preds).view(-1,3)
        train_auroc=self.auroc(all_preds.float()[:,1],all_preds.long()[:,-1])
        train_auroc_gather=self.auroc(all_preds_gather.float()[:,1],all_preds_gather.long()[:,-1])
        train_loss=self.ce_loss(all_preds[:,:-1],all_preds.long()[:,-1])
        self.log('train_loss',train_loss,sync_dist=True)
        self.log('train_auroc_gathered',train_auroc_gather)
        if self.trainer.global_rank==0:
            print('\n------gathered auroc is %s----\n'%train_auroc_gather,flush=True)
        del all_preds, train_auroc,train_loss
        self.train_out.clear()

    def validation_step(self, batch, batch_idx):
        self.esm_model.eval()
        self.proj.eval()
        torch.cuda.empty_cache()
        labels=batch['label'].long()
        locs=batch['Loc'].long()

        if self.crop_val is False: #no cropping
            print('validation is not cropped')
            seqs=batch['seq']
            mutated_batch_samples=list(zip(locs,seqs))
            wild_batch_samples=self.get_wild_batch(batch,starts=[0]*len(seqs),pos=None)
        else:
            seqs,starts,pos=self.crop_batch(batch,batch_idx)
            batch['seq']=seqs
            len_seqs=[len(seq) for seq in seqs]
            # print('===================================\nlength of seqs in this batch(%s) is %s\n========================='%(batch_idx,len_seqs),flush=True)
            mutated_batch_samples=list(zip(locs,seqs))
            wild_batch_samples=self.get_wild_batch(batch,starts=starts,pos=pos)

            self.print_memory('initiated batches',detail=True)
        del seqs

        try:
            mutated_embs=self.get_esm_embedings(mutated_batch_samples,loc=locs,starts=starts,local_range=self.local_range)
            self.print_memory('mutated_embs obtained',detail=True)

            wild_embs=self.get_esm_embedings(wild_batch_samples,loc=locs,starts=starts,local_range=self.local_range)
            self.print_memory('wild_embs obtained',detail=True)
        except AttributeError:
            print('\n',seqs,'\n',mutated_batch_samples,'\n',wild_batch_samples,'\n')
            print(batch)
            return 0
        batch_size=len(labels)

        del mutated_batch_samples,wild_batch_samples,batch

        embs=self.define_embds(wild_embs,mutated_embs)
        y=self.proj(embs.float().to(self.device))
        self.print_memory('after projection',detail=True)

        del mutated_embs,wild_embs,embs

        pred=torch.hstack([y,labels.reshape(batch_size,1)]).cpu()
        val_loss=self.ce_loss(y,labels.long())
        # print(y,labels)
        # print(val_loss)

        self.val_out.append(pred)
        return pred
    def random_crop(self,seq,pos):
        np.random.seed(int('%d%d'%(self.trainer.current_epoch,self.trainer.global_step)))
        right=len(seq)-self.crop_len
        left=0
        min_start=max(left,pos-self.crop_len+1)
        max_start=min(right,pos)
        if pos>=len(seq):start=len(seq)-self.crop_len
        else: start=np.random.randint(low=min_start,high=max_start+1)
        seq_after=seq[start:start+self.crop_len]
        return seq_after


    def center_crop(self,seq,pos):
        left_half=self.crop_len//2
        right_half=self.crop_len-left_half
        ideal_right=pos+right_half
        ideal_left=pos-left_half

        actual_right=len(seq)
        actual_left=0
        if actual_left>ideal_left:left,right=actual_left,actual_left+self.crop_len
        else:
            if actual_right<ideal_right:left,right=actual_right-self.crop_len,actual_right
            else:left,right=ideal_left,ideal_right
        seq_after=seq[left:right]
        return seq_after,left
    def on_validation_epoch_end(self):
        all_preds=torch.vstack(self.val_out)
        all_preds_gather=self.all_gather(all_preds).view(-1,3)
        val_auroc=self.auroc(all_preds.float()[:,1],all_preds.long()[:,-1])
        val_auprc_gather=self.auprc(all_preds_gather.float()[:,1],all_preds_gather.long()[:,-1])
        
        val_auroc_gather=self.auroc(all_preds_gather.float()[:,1],all_preds_gather.long()[:,-1])
        val_loss=self.ce_loss(all_preds[:,:-1],all_preds.long()[:,-1])
        self.log('val_loss',val_loss,sync_dist=True)
        self.log('val_auroc_gathered',val_auroc_gather)
        self.log('val_auprc_gathered',val_auprc_gather)
        val_auroc_average=torch.mean(self.all_gather(val_auroc),dim=0)
        if self.trainer.global_rank==0:
            print('gathered auroc is %s'%val_auroc_gather)
            print('gathered auprc is %s'%val_auprc_gather)
        del all_preds, val_auroc,val_loss
        self.val_out.clear()

    def test_step(self, batch, batch_idx):
        self.esm_model.eval()
        self.proj.eval()
        torch.cuda.empty_cache()
        locs=batch['Loc'].long()

        labels=batch['label'].long()
        if self.crop_val is False: #no cropping
            print('test is not cropped')
            seqs=batch['seq']
            mutated_batch_samples=list(zip(locs,seqs))
            wild_batch_samples=self.get_wild_batch(batch,starts=[0]*len(seqs),pos=None)
        else:
            seqs,starts,pos=self.crop_batch(batch,batch_idx)
            batch['seq']=seqs
            len_seqs=[len(seq) for seq in seqs]
            # print('===================================\nlength of seqs in this batch(%s) is %s\n========================='%(batch_idx,len_seqs),flush=True)
            mutated_batch_samples=list(zip(locs,seqs))
            wild_batch_samples=self.get_wild_batch(batch,starts=starts,pos=pos)

            self.print_memory('initiated batches',detail=True)
        del seqs

        try:
            mutated_embs=self.get_esm_embedings(mutated_batch_samples,starts=starts,loc=locs,local_range=self.local_range)
            self.print_memory('mutated_embs obtained',detail=True)

            wild_embs=self.get_esm_embedings(wild_batch_samples,starts=starts,loc=locs,local_range=self.local_range)
            self.print_memory('wild_embs obtained',detail=True)
        except AttributeError:
            print('\n',seqs,'\n',mutated_batch_samples,'\n',wild_batch_samples,'\n')
            print(batch)
            return 0
        batch_size=len(labels)

        del mutated_batch_samples,wild_batch_samples,batch
        #TODO: multichannel mlp


        embs=self.define_embds(wild_embs,mutated_embs)


        y=self.proj(embs.float().to(self.device))
        self.print_memory('after projection',detail=True)

        del mutated_embs,wild_embs,embs

        pred=torch.hstack([y,labels.reshape(batch_size,1)]).cpu()
        self.test_out.append(pred)
        return pred

    def on_test_epoch_end(self):
        all_preds=torch.vstack(self.test_out)
        all_preds_gather=self.all_gather(all_preds).view(-1,3)
        test_auroc_gather=self.auroc(all_preds_gather.float()[:,1],all_preds_gather.long()[:,-1])
        test_auprc_gather=self.auprc(all_preds_gather.float()[:,1],all_preds_gather.long()[:,-1])
        self.log('test_auroc_gathered',test_auroc_gather)
        self.log('test_auprc_gathered',test_auprc_gather)
        if self.trainer.global_rank==0:
            print('gathered auroc is %s'%test_auroc_gather)
            print('gathered auprc is %s'%test_auprc_gather)
        del all_preds
        self.test_out.clear()
    
    def get_esm_embedings(self,batch_sample,loc=None,local_range=0,starts=None):
        torch.cuda.empty_cache()
        mutation_locs, _, batch_tokens=self.batch_converter(batch_sample)
        for i in range(len(mutation_locs)):
            if starts[i]:
                mutation_locs[i]=int(mutation_locs[i])-int(starts[i])
        batch_tokens=batch_tokens.to(self.device)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        results = self.esm_model(batch_tokens, repr_layers=[self.repr_layers], return_contacts=False)

        token_representations = results["representations"][self.repr_layers]


        del results
        sequence_representations = []
        local_representations=[]
        AA_representations=[]
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))    
            # print(max(1,mutation_locs[i]-local_range+1),min(batch_lens[i],mutation_locs[i]+local_range+2))    
            local_representations.append(token_representations[i,max(1,mutation_locs[i]-local_range+1):min(batch_lens[i],mutation_locs[i]+local_range+1)].mean(0))
            # print(local_representations)
            AA_representations.append(token_representations[i,mutation_locs[i]+1])
        
        del token_representations
        torch.cuda.empty_cache()
        sequence_representations=torch.vstack(sequence_representations)
        local_representations=torch.vstack(local_representations)
        AA_representations=torch.vstack(AA_representations)
        del batch_lens,batch_tokens
        return {'sequence_representations':sequence_representations,
                'local_representations':local_representations,
                'AA_representations':AA_representations
        }

    def get_current_epoch_max_len(self):
        if self.trainer.which_dl=='short':
            return self.trainer.datamodule.max_short
        elif self.trainer.which_dl=='medium':
            return self.trainer.datamodule.max_medium
        elif self.trainer.which_dl=='long':
            return self.trainer.datamodule.max_long

    def get_wild_batch(self,mutated_batch,starts=None,pos=None):
        uniprots=mutated_batch['UniProt']
        mutants=mutated_batch['seq']
        locs=mutated_batch['Loc']
        mutant_lens=[len(seq) for seq in mutants]
        seqs=[get_sequence_from_uniprot_id(uniprot) for uniprot in uniprots]
        batch_sample=[]
        lens=[]
        for i,seq in enumerate(seqs):

            if starts[i]==0: #no random cropping
                # print('this seq is not cropped as the length is %s, starts[i] is %s'%(len(seq),starts[i]))
                lens.append(mutant_lens[i])
                batch_sample.append((locs[i],seq[:mutant_lens[i]])) #the wild seq has to be of same length as the mutant

            elif starts[i] is not None: #if mutated sequences are cropped and a start is returned
                seq=seq[starts[i]:starts[i]+self.crop_len]
                batch_sample.append((locs[i],seq))
                lens.append(len(seq))

            elif starts[i] is None and len(seq)>self.crop_len: # if mutated seq is not cropped but the wild is too long
                seq,_=self.crop(seq,pos[i])
                batch_sample.append((locs[i],seq))
                lens.append(len(seq))
            else:
                batch_sample.append((locs[i],seq))
                lens.append(len(seq))
            
                
        # print('\n length of wild batch is %s'%lens)
        return batch_sample


class Esm_delta_multiscale_weight(Esm_finetune_delta):
    def __init__(self,esm_model,esm_model_dim,repr_layers,lr,unfreeze_n_layers,num_bins=8,bin_one_side_distance=[0,2,4,8,16,32,128,256],save_embeddings=False):
        self.num_bins=num_bins
        super().__init__(esm_model=esm_model,esm_model_dim=esm_model_dim,repr_layers=repr_layers,lr=lr,unfreeze_n_layers=unfreeze_n_layers)
        self.bin_one_side_distance=bin_one_side_distance
        self.Weights=nn.ModuleList([WeightModule(2*bin_one_side_distance[i]+1) for i in range(num_bins)])
        self.proj=nn.Sequential(
            nn.Linear(self.get_embs_dim(),2),
            nn.Softmax(dim=1)
        )
        self.batch_sample='random'
        self.init_dataset()
        self.auroc=BinaryAUROC()
        self.auprc=AveragePrecision(task='binary')
       
        self.save_embeddings=save_embeddings
        self.val_embds={}
        self.test_embds={}

        self.val_out=[]
        self.test_out=[]
        self.train_out=[]
        
        self.crop=self.center_crop
        self.crop_batch=self.center_crop_batch
        self.crop_val=True
        
        self.ce_loss=torch.nn.CrossEntropyLoss()
        print('Using Normal CE Loss')

    def get_cropped_batch_samples(self,batch,batch_idx):
        labels=batch['label'].long()
        locs=batch['Loc'].long()
        seqs,starts,pos=self.crop_batch(batch,batch_idx)
        batch['seq']=seqs
        mutated_batch_samples=list(zip(locs,seqs))
        wild_batch_samples=self.get_wild_batch(batch,starts=starts,pos=pos) 
        return starts,labels,locs,mutated_batch_samples,wild_batch_samples
    
    def get_embs_dim(self):
        return (self.num_bins+1)*self.esm_model_dim*2 #sequence representation + bin_representations for both delta and wild

    def get_esm_embedings(self,batch_sample,starts=None):
        mutation_locs, _, batch_tokens=self.batch_converter(batch_sample)
        for i in range(len(mutation_locs)):
            if starts[i]:
                mutation_locs[i]=int(mutation_locs[i])-int(starts[i])
        batch_tokens=batch_tokens.to(self.device)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        results = self.esm_model(batch_tokens, repr_layers=[self.repr_layers], return_contacts=False)

        token_representations = results["representations"][self.repr_layers]

        del results
        
        bin_representations=[]
        sequence_representations=[]
        for i in range(len(batch_lens)):
            l=[] 
            #first token is the <cls>
            for j,(_,one_side_len) in enumerate(self.multiscale_bins()):
                left=max(1,mutation_locs[i]-one_side_len+1)
                right=min(batch_lens[i],mutation_locs[i]+one_side_len+2)
                weight_left_distance=one_side_len-mutation_locs[i] if left==1 else 0
                weight_right_distance=mutation_locs[i]+one_side_len+2-batch_lens[i] if right==batch_lens[i] else 0
                l.append(torch.matmul(self.Weights[j](range(weight_left_distance,2*one_side_len+1-weight_right_distance)).T,token_representations[i,left:right])) #TODO: normalize weight
            bin_representations.append(torch.hstack(l))
            sequence_representations.append(token_representations[i, 1: batch_lens[i] - 1].mean(0))    

        del token_representations
        torch.cuda.empty_cache()
        sequence_representations=torch.vstack(sequence_representations)
        bin_representations=torch.vstack(bin_representations)
        del batch_lens,batch_tokens
        return {'sequence_representations':sequence_representations,
                'bin_representations':bin_representations,
        }
    
    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        starts,labels,locs,mutated_batch_samples,wild_batch_samples=self.get_cropped_batch_samples(batch,batch_idx)
        mutated_embs=self.get_esm_embedings(mutated_batch_samples,starts=starts)
        wild_embs=self.get_esm_embedings(wild_batch_samples,starts=starts)

        embs=self.define_embds(wild_embs,mutated_embs)
        y=self.proj(embs.float().to(self.device))

        del mutated_embs,wild_embs,embs

        loss=self.ce_loss(y,labels)
        self.train_out.append(torch.hstack([y,labels.reshape(len(labels),1)]).cpu())

        del y,labels

        return loss

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



    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        starts,labels,locs,mutated_batch_samples,wild_batch_samples=self.get_cropped_batch_samples(batch,batch_idx)
        mutated_embs=self.get_esm_embedings(mutated_batch_samples,starts=starts)
        wild_embs=self.get_esm_embedings(wild_batch_samples,starts=starts)

        embs=self.define_embds(wild_embs,mutated_embs)
        if self.save_embeddings:
            for i in range(embs.shape[0]):
                self.val_embds[batch['Name'][i]]={'embs':embs[i,:].cpu(),'UniProt':batch['UniProt'][i],'Loc':locs[i],'label':labels[i]}
                
        y=self.proj(embs.float().to(self.device))

        del mutated_embs,wild_embs,embs

        loss=self.ce_loss(y,labels)
        self.val_out.append(torch.hstack([y,labels.reshape(len(labels),1)]).cpu())

        del y,labels

        return loss

    def on_validation_epoch_end(self) :
        all_preds=torch.vstack(self.val_out)
        all_preds_gather=self.all_gather(all_preds).view(-1,3)
        val_auroc_gather=self.auroc(all_preds_gather.float()[:,1],all_preds_gather.long()[:,-1])
        val_auprc_gather=self.auprc(all_preds_gather.float()[:,1],all_preds_gather.long()[:,-1])
        val_loss=self.ce_loss(all_preds[:,:-1],all_preds.long()[:,-1])
        # self.log('val_loss',val_loss,sync_dist=True)
        # self.log('val_auroc_gathered',val_auroc_gather)
        # self.log('val_auprc_gathered',val_auprc_gather)
        if self.trainer.global_rank==0:
            print('\n------gathered auroc is %s----\n'%val_auroc_gather,flush=True)
            print('\n------gathered auprc is %s----\n'%val_auprc_gather,flush=True)
        del all_preds, val_auroc_gather, val_auprc_gather,val_loss
        self.val_out.clear()
        if self.val_embds:
            torch.save(self.val_embds,'/scratch/user/zshuying/ppi_mutation/data/baseline0/val_embds_%s.pt'%self.trainer.global_rank)

    def test_step(self, batch, batch_idx,save_embeddings=False):
        torch.cuda.empty_cache()
        starts,labels,locs,mutated_batch_samples,wild_batch_samples=self.get_cropped_batch_samples(batch,batch_idx)
        mutated_embs=self.get_esm_embedings(mutated_batch_samples,starts=starts)
        wild_embs=self.get_esm_embedings(wild_batch_samples,starts=starts)

        embs=self.define_embds(wild_embs,mutated_embs)
        y=self.proj(embs.float().to(self.device))

        del mutated_embs,wild_embs,embs

        loss=self.ce_loss(y,labels)
        self.test_out.append(torch.hstack([y,labels.reshape(len(labels),1)]).cpu())

        del y,labels

        return loss

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


    def define_embds(self,wild_embs,mutated_embs):

        wild_mean_embs,wild_bin_embs=wild_embs['sequence_representations'],wild_embs['bin_representations']
        mutated_mean_embs,mutated_bin_embs=mutated_embs['sequence_representations'],mutated_embs['bin_representations']
        delta_mean_embs=mutated_mean_embs-wild_mean_embs
        delta_bin_embs=mutated_bin_embs-wild_bin_embs
        l=[wild_mean_embs,wild_bin_embs,delta_mean_embs,delta_bin_embs]
        #TODO layernorm?
        return torch.hstack(l)
    

    

    


    def multiscale_bins(self):
        """
        multiscale_bins=[([],bin_oneside_distance),([],bin_oneside_distance)]
        """
        multiscale_bins=[([],i) for i in self.bin_one_side_distance]
        return multiscale_bins



class Esm_multiscale_wildonly(Esm_delta_multiscale_weight):
    def __init__(self,esm_model,esm_model_dim,repr_layers,lr,unfreeze_n_layers,num_bins=8,bin_one_side_distance=[0,2,4,8,16,32,128,256],save_embeddings=True):
        super().__init__(esm_model,esm_model_dim,repr_layers,lr,unfreeze_n_layers,num_bins,bin_one_side_distance,True)
    
    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        uniprots,random_cropped_wild=self.random_crop_wild(batch)
        wild_embs=self.get_esm_embedings(random_cropped_wild)
        embs=self.define_embds(wild_embs)
        for i in range(embs.shape[0]):
            self.val_embds[batch['UniProt'][i]]={'embs':embs[i,:].cpu()}
        print('working')
        y=self.proj(embs.float().to(self.device))
        return y


    def define_embds(self,wild_embs):
        wild_mean_embs,wild_bin_embs=wild_embs['sequence_representations'],wild_embs['bin_representations']
        delta_mean_embs,delta_bin_embs=torch.zeros_like(wild_mean_embs).to(wild_mean_embs),torch.zeros_like(wild_bin_embs.to(wild_mean_embs))
        l=[wild_mean_embs,wild_bin_embs,delta_mean_embs,delta_bin_embs]
        # for item in l:
        #     print(item.device)
        return torch.hstack(l)
    
    def random_crop_wild(self,batch):
        # print(batch)
        seqs=batch['seq']
        # print(seqs)
        uniprots=batch['UniProt']
        batch_after=[]
        for i in range(len(seqs)):
            seq=seqs[i]
            uniprot=uniprots[i]
            try:seq_len=len(seq)
            except TypeError: continue
            if seq_len>1500:
                left_max=seq_len-1500
                start=np.random.randint(0,left_max+1)
            else:
                start=0
            batch_after.append((uniprot,seq[start:start+1500]))
        return uniprots,batch_after
    def on_validation_epoch_end(self) :
        torch.save(self.val_embds,'/scratch/user/zshuying/ppi_mutation/data/baseline0/all_wild_embeddings_%s.pt'%self.trainer.global_rank)
     

    
    def get_esm_embedings(self,batch_sample):
        _, _, batch_tokens=self.batch_converter(batch_sample)
        batch_tokens=batch_tokens.to(self.device)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        results = self.esm_model(batch_tokens, repr_layers=[self.repr_layers], return_contacts=False)

        token_representations = results["representations"][self.repr_layers]

        del results
        
        bin_representations=[]
        sequence_representations=[]
        for i in range(len(batch_lens)):
            l=[] 
            for j,(_,one_side_len) in enumerate(self.multiscale_bins()):
                # left=max(1,mutation_locs[i]-one_side_len+1)
                # right=min(batch_lens[i],mutation_locs[i]+one_side_len+2)
                # weight_left_distance=one_side_len-mutation_locs[i] if left==1 else 0
                # weight_right_distance=mutation_locs[i]+one_side_len+2-batch_lens[i] if right==batch_lens[i] else 0
                # l.append(torch.matmul(self.Weights[j](range(weight_left_distance,2*one_side_len+1-weight_right_distance)).T,token_representations[i,left:right]))
                l.append(torch.zeros((1,self.esm_model_dim)))
            bin_representations.append(torch.hstack(l))
            sequence_representations.append(token_representations[i, 1: batch_lens[i] - 1].mean(0))    

        del token_representations
        torch.cuda.empty_cache()
        sequence_representations=torch.vstack(sequence_representations)
        bin_representations=torch.vstack(bin_representations)
        del batch_lens,batch_tokens
        return {'sequence_representations':sequence_representations,
                'bin_representations':bin_representations.to(sequence_representations),
        }

class WeightModule(pl.LightningModule):
    def __init__(self,num_weights):
        super(WeightModule, self).__init__()
        self.num_weights = num_weights
        self.W = nn.Parameter(torch.ones(self.num_weights, 1))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, indices):
        selected_weights = self.W[indices]
        normalized_weights = self.softmax(selected_weights)
        return normalized_weights



class ESM_finetune_relative(Esm_finetune_delta):
    def __init__(self,num_heads=4,ffn_embed_dim=1280,
                 esm_model=esm.pretrained.esm2_t12_35M_UR50D(),crop_val=False,esm_model_dim=480,n_class=2,truncation_len=None,unfreeze_n_layers=3,repr_layers=12,batch_sample='random',which_embds=0,lr=None,crop_len=None,debug=False,crop_mode='random',balanced_loss=False,local_range=0):
        super().__init__(esm_model=esm_model,crop_val=crop_val,esm_model_dim=esm_model_dim, n_class=n_class,truncation_len=truncation_len,unfreeze_n_layers=unfreeze_n_layers,repr_layers=repr_layers,batch_sample=batch_sample,which_embds=which_embds,lr=lr,crop_len=crop_len,crop_mode=crop_mode,balanced_loss=balanced_loss,local_range=local_range)


class ESM_finetune_ape(Esm_finetune_delta):
    def __init__(self,
                 esm_model=esm.pretrained.esm2_t12_35M_UR50D(),crop_val=False,esm_model_dim=480,n_class=2,truncation_len=None,unfreeze_n_layers=3,repr_layers=12,batch_sample='random',which_embds=0,lr=None,crop_len=None,debug=False,crop_mode='random',balanced_loss=False,local_range=0):
        super().__init__(esm_model=esm_model,crop_val=crop_val,esm_model_dim=esm_model_dim, n_class=n_class,truncation_len=truncation_len,unfreeze_n_layers=unfreeze_n_layers,repr_layers=repr_layers,batch_sample=batch_sample,which_embds=which_embds,lr=lr,crop_len=crop_len,crop_mode=crop_mode,balanced_loss=balanced_loss,local_range=local_range)
        self.embed_positions=esm.modules.LearnedPositionalEmbedding(local_range*2+1,esm_model_dim,None)

    





# import torch
# import gc
# for obj in gc.get_objects():
#     try:
#         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
#             mem=obj.element_size()*obj.nelement()/1024**3
#             if mem>0.5:
#                 print(obj, obj.size(),mem)
#     except: pass


class RelativePosition(pl.LightningModule):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table[final_mat].cuda()

        return embeddings

class RelativeMultiHeadAttentionLayer(pl.LightningModule):
    def __init__(self, hid_dim, n_heads):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.max_relative_position = 2

        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        
    def forward(self, query, key, value, mask = None):
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2)) 

        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size*self.n_heads, self.head_dim)
        r_k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)
        attn = (attn1 + attn2) / self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)

        attn = torch.softmax(attn, dim = -1)

        #attn = [batch size, n heads, query len, key len]
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size*self.n_heads, len_k)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.head_dim)

        x = weight1 + weight2
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x



class RelativeTransformerLayer(pl.LightningModule):
    def __init__(self, hid_dim, n_heads,ffn_embed_dim) -> None:
        super().__init__()
        self.embed_dim=hid_dim
        self.relative_attention=RelativeMultiHeadAttentionLayer(hid_dim,n_heads)
        self.self_attn_layer_norm=esm.modules.ESM1LayerNorm(hid_dim)
        self.ffn_embed_dim=ffn_embed_dim
        #only local delta
        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = esm.modules.ESM1LayerNorm(self.embed_dim)



class Attention_Weight(nn.Module):
    def __init__(self, input_dim):
        super(Attention_Weight, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted
    





