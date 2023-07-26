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
from balanced_loss import Loss

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
    def __init__(self, esm_model=esm.pretrained.esm2_t36_3B_UR50D(),esm_model_dim=2560,truncation_len=None,unfreeze_n_layers=10,repr_layers=36,lr=4*1e-3,crop_len=None):
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
    


class Esm_finetune_delta(Esm_finetune):
    def __init__(self, esm_model=esm.pretrained.esm2_t12_35M_UR50D(),crop_val=False,esm_model_dim=480,n_class=2,truncation_len=None,unfreeze_n_layers=3,repr_layers=12,batch_sample='random',which_embds=0,lr=None,crop_len=None,debug=False,crop_mode='random',balanced_loss=False):
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
        self.init_dataset()
        self.auroc=BinaryAUROC()
        self.auprc=AveragePrecision(task='binary')
        self.embs_dim=len(str(which_embds))*self.esm_model_dim
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
            self.ce_loss=nn.functional.cross_entropy
            print('Using Normal CE Loss')

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
        seqs,starts,pos=self.crop_batch(batch,batch_idx)
        batch['seq']=seqs
        len_seqs=[len(seq) for seq in seqs]
        # print('===================================\nlength of seqs in this batch(%s) is %s\n========================='%(batch_idx,len_seqs),flush=True)
        mutated_batch_samples=list(zip(labels,seqs))
        del seqs
        wild_batch_samples=self.get_wild_batch(batch,starts=starts,pos=pos)
        del batch
        self.print_memory('initiated batches',detail=True)

        try:
            mutated_embs=self.get_esm_embedings(mutated_batch_samples)
            self.print_memory('mutated_embs obtained',detail=True)

            wild_embs=self.get_esm_embedings(wild_batch_samples)
            self.print_memory('wild_embs obtained',detail=True)
        except :
            # print('\n',mutated_batch_samples,'\n',wild_batch_samples,'\n')
            print([len(sample[1]) for sample in wild_batch_samples])
            return 0
            
        batch_size=wild_embs.shape[0]

        del mutated_batch_samples,wild_batch_samples

        delta_embs=mutated_embs-wild_embs

        embs=self.define_embds(delta_embs,wild_embs,mutated_embs)

        y=self.proj(embs.float().to(self.device))

        del delta_embs,mutated_embs,wild_embs,embs

        loss=self.ce_loss(y,labels)
        torch.cuda.empty_cache()
        self.train_out.append(torch.hstack([y,labels.reshape(batch_size,1)]).cpu())

        del y,labels

        return loss
    

    def define_embds(self,delta_embds,wild_embs,mutated_embs):
        dict_embds={
            '0':delta_embds,
            '1':wild_embs,
            '2':mutated_embs,
        }
        l=[]
        for i in str(self.which_embds):
            l.append(dict_embds[i])
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
        if self.crop_val is False: #no cropping
            print('validation is not cropped')
            seqs=batch['seq']
            mutated_batch_samples=list(zip(labels,seqs))
            wild_batch_samples=self.get_wild_batch(batch,starts=[0]*len(seqs),pos=None)
        else:
            seqs,starts,pos=self.crop_batch(batch,batch_idx)
            batch['seq']=seqs
            len_seqs=[len(seq) for seq in seqs]
            # print('===================================\nlength of seqs in this batch(%s) is %s\n========================='%(batch_idx,len_seqs),flush=True)
            mutated_batch_samples=list(zip(labels,seqs))
            wild_batch_samples=self.get_wild_batch(batch,starts=starts,pos=pos)

            self.print_memory('initiated batches',detail=True)
        del seqs

        try:
            mutated_embs=self.get_esm_embedings(mutated_batch_samples)
            self.print_memory('mutated_embs obtained',detail=True)

            wild_embs=self.get_esm_embedings(wild_batch_samples)
            self.print_memory('wild_embs obtained',detail=True)
        except AttributeError:
            print('\n',seqs,'\n',mutated_batch_samples,'\n',wild_batch_samples,'\n')
            print(batch)
            return 0
        batch_size=wild_embs.shape[0]

        del mutated_batch_samples,wild_batch_samples,batch
        #TODO: multichannel mlp
        self.print_memory('before delta',detail=True)

        delta_embs=mutated_embs-wild_embs
        self.print_memory('after delta',detail=True)

        embs=self.define_embds(delta_embs,wild_embs,mutated_embs)


        y=self.proj(embs.float().to(self.device))
        self.print_memory('after projection',detail=True)

        del delta_embs,mutated_embs,wild_embs,embs

        pred=torch.hstack([y,labels.reshape(batch_size,1)]).cpu()
        self.val_out.append(pred)
        return pred

    def on_validation_epoch_end(self):
        all_preds=torch.vstack(self.val_out)
        all_preds_gather=self.all_gather(all_preds).view(-1,3)
        val_auroc=self.auroc(all_preds.float()[:,1],all_preds.long()[:,-1])
        val_auroc_gather=self.auroc(all_preds_gather.float()[:,1],all_preds_gather.long()[:,-1])
        val_loss=self.ce_loss(all_preds[:,:-1],all_preds.long()[:,-1])
        self.log('val_loss',val_loss,sync_dist=True)
        self.log('val_auroc_gathered',val_auroc_gather)
        val_auroc_average=torch.mean(self.all_gather(val_auroc),dim=0)
        if self.trainer.global_rank==0:
            print('gathered auroc is %s'%val_auroc_gather)
        del all_preds, val_auroc,val_loss
        self.val_out.clear()

    def test_step(self, batch, batch_idx):
        self.esm_model.eval()
        self.proj.eval()
        torch.cuda.empty_cache()
        labels=batch['label'].long()
        if self.crop_val is False: #no cropping
            print('test is not cropped')
            seqs=batch['seq']
            mutated_batch_samples=list(zip(labels,seqs))
            wild_batch_samples=self.get_wild_batch(batch,starts=[0]*len(seqs),pos=None)
        else:
            seqs,starts,pos=self.crop_batch(batch,batch_idx)
            batch['seq']=seqs
            len_seqs=[len(seq) for seq in seqs]
            # print('===================================\nlength of seqs in this batch(%s) is %s\n========================='%(batch_idx,len_seqs),flush=True)
            mutated_batch_samples=list(zip(labels,seqs))
            wild_batch_samples=self.get_wild_batch(batch,starts=starts,pos=pos)

            self.print_memory('initiated batches',detail=True)
        del seqs

        try:
            mutated_embs=self.get_esm_embedings(mutated_batch_samples)
            self.print_memory('mutated_embs obtained',detail=True)

            wild_embs=self.get_esm_embedings(wild_batch_samples)
            self.print_memory('wild_embs obtained',detail=True)
        except AttributeError:
            print('\n',seqs,'\n',mutated_batch_samples,'\n',wild_batch_samples,'\n')
            print(batch)
            return 0
        batch_size=wild_embs.shape[0]

        del mutated_batch_samples,wild_batch_samples,batch
        #TODO: multichannel mlp
        self.print_memory('before delta',detail=True)

        delta_embs=mutated_embs-wild_embs
        self.print_memory('after delta',detail=True)

        embs=self.define_embds(delta_embs,wild_embs,mutated_embs)


        y=self.proj(embs.float().to(self.device))
        self.print_memory('after projection',detail=True)

        del delta_embs,mutated_embs,wild_embs,embs

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
    
    def get_esm_embedings(self,batch_sample):
        torch.cuda.empty_cache()
        _, _, batch_tokens=self.batch_converter(batch_sample)
        batch_tokens=batch_tokens.to(self.device)
        # batch_labels=torch.stack(batch_labels)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        self.print_memory('inside get_esm_embeddings (1)',detail=True)
        results = self.esm_model(batch_tokens, repr_layers=[self.repr_layers], return_contacts=False)
        self.print_memory('inside get_esm_embeddings (2)',detail=True)

        token_representations = results["representations"][self.repr_layers]
        self.print_memory('inside get_esm_embeddings (3)',detail=True)

        del results
        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))
        self.print_memory('inside get_esm_embeddings (4)',detail=True)
        
        del token_representations
        torch.cuda.empty_cache()
        sequence_representations=torch.stack(sequence_representations)
        self.print_memory('inside get_esm_embeddings (5)',detail=True)

        del batch_lens,batch_tokens
        return sequence_representations

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
        mutant_lens=[len(seq) for seq in mutants]
        seqs=[get_sequence_from_uniprot_id(uniprot) for uniprot in uniprots]
        batch_sample=[]
        lens=[]
        for i,seq in enumerate(seqs):

            if starts[i]==0: #no random cropping
                # print('this seq is not cropped as the length is %s, starts[i] is %s'%(len(seq),starts[i]))
                lens.append(mutant_lens[i])
                batch_sample.append((uniprots[i],seq[:mutant_lens[i]])) #the wild seq has to be of same length as the mutant

            elif starts[i] is not None: #if mutated sequences are cropped and a start is returned
                seq=seq[starts[i]:starts[i]+self.crop_len]
                batch_sample.append((uniprots[i],seq))
                lens.append(len(seq))

            elif starts[i] is None and len(seq)>self.crop_len: # if mutated seq is not cropped but the wild is too long
                seq,_=self.crop(seq,pos[i])
                batch_sample.append((uniprots[i],seq))
                lens.append(len(seq))
            else:
                batch_sample.append((uniprots[i],seq))
                lens.append(len(seq))
            
                
        # print('\n length of wild batch is %s'%lens)
        return batch_sample

        



# import torch
# import gc
# for obj in gc.get_objects():
#     try:
#         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
#             mem=obj.element_size()*obj.nelement()/1024**3
#             if mem>0.5:
#                 print(obj, obj.size(),mem)
#     except: pass
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