import torch
from torch.utils.data import Dataset
global top_path  # the path of the top_level directory
global script_path, data_path, logging_path
from transformers import AutoTokenizer, EsmForSequenceClassification
import os, sys
from torch.utils.data import DataLoader
from torch import optim, nn, utils, Tensor
import lightning.pytorch as pl
import esm
import torch.cuda as cuda

from torchmetrics.classification import BinaryAUROC, MulticlassAUROC
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
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import BasePredictionWriter

top_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(find_current_path()))))
sys.path.append(top_path)
from scripts.utils import *
from scripts.myesm.datasets import ProteinSequence
class hfESM(pl.LightningModule):
    def __init__(self,model="esm2_t6_8M_UR50D",num_labels=3):
        super().__init__()
        self.model=EsmForSequenceClassification.from_pretrained("facebook/"+model, num_labels=num_labels)
        self.model.max_position_embeddings=2048
        self.num_labels=num_labels
        self.esm_model=model
        self.tokenizer=AutoTokenizer.from_pretrained("facebook/"+model)

    # def forward(self,**inputs):
    #     return self.model(**inputs)
    
    def training_step(self,batch,batch_idx):
        labels,seqs=batch['label'].long(),batch['seq']
        seqs=self.tokenizer(seqs,padding=True,  return_tensors="pt").input_ids.to(labels)
        loss=self.model(seqs,labels=labels).loss
        self.log('train_loss',loss,on_step=True,on_epoch=True,sync_dist=True)
        return loss

    def validation_step(self,batch,batch_idx):
        labels,seqs=batch['label'].long(),batch['seq']
        seqs=self.tokenizer(seqs,padding=True,  return_tensors="pt").input_ids.to(labels)
        results=self.model(seqs,labels=labels)
        loss,logits=results['loss'],results['logits']
        preds=torch.argmax(logits,axis=1)
        self.log('val_loss',loss,on_step=True,on_epoch=True,sync_dist=True)
        return {"loss":loss, "preds":preds, "labels":labels}

    def configure_optimizers(self,lr=1e-8) :
        optimizer=optim.Adam(self.parameters(),lr=lr)
        return optimizer


class myMLP(pl.LightningModule):
    def __init__(self,input_dim,hidden_dim1=320,hidden_dim2=420,learning_rate=1e-3):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(input_dim,hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1,hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2,2),
            nn.Softmax(dim=1)
        )
        self.learning_rate=learning_rate

        self.metrics=BinaryAUROC(thresholds=None)
    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        self.layers.train()
        #training loop definition
        labels,embeddings=batch['label'].long(),batch['embedding']
        y=self.layers(embeddings)
        loss=nn.functional.cross_entropy(y,labels)

        self.log('train_loss',loss)
        torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        self.layers.eval()
        #training loop definition
        labels,embeddings=batch['label'].long(),batch['embedding']
        y=self.layers(embeddings)
        loss=nn.functional.cross_entropy(y,labels)
        auroc=self.metrics(y,labels)
        self.log('val_loss',loss)
        self.log('val_auroc',auroc)
        torch.cuda.empty_cache()
        return loss


    def configure_optimizers(self) :
        optimizer=optim.Adam(self.parameters(), self.learning_rate)
        return optimizer

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

        idxes,labels,seqs=batch['idx'],batch['label'].long(),batch['seq']
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
            results = self.esm_model(batch_tokens, repr_layers=[34], return_contacts=False)
            token_representations = results["representations"][34]
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
    #     torch.save({"prediction":prediction,"label":batch['label'].long()}, os.path.join(self.output_dir, dataloader_idx, f"{batch_idx}.pt"))
        


class Esm_mlp(pl.LightningModule):
    def __init__(self, mlp_input_dim, mlp_hidden_dim, esm_model=esm.pretrained.esm2_t6_8M_UR50D(),truncation_len=None,mixed_cpu=True ):
        super().__init__()
        self.save_hyperparameters(ignore=['esm_model','mlp','alphabet'], logger=True)
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
        idxes,labels,seqs=batch['idx'],batch['label'].long(),batch['seq']
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
        idxes,labels,seqs=batch['idx'],batch['label'].long(),batch['seq']
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
        idxes,labels,seqs=batch['idx'],batch['label'].long(),batch['seq']
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




class Esm_finetune(pl.LightningModule):
    def __init__(self, esm_model=esm.pretrained.esm2_t36_3B_UR50D(),esm_model_dim=2560,truncation_len=None,unfreeze_n_layers=10,repr_layers=36,lr=4*1e-3,random_crop_len=None):
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
        self.random_crop_len=random_crop_len


    def random_crop_batch(self,batch,batch_idx):
        names,seqs=batch['Name'],batch['seq']
        seqs_after=[]
        starts=[]
        positions=[]
        for i in range(len(batch['Name'])):
            name=names[i]
            seq=seqs[i]
            pos=self.get_pos_of_name(name)-1 #it counts from 1 in biology instead 0 in python
            if len(seq)>self.random_crop_len:
                np.random.seed(int('%d%d'%(self.trainer.current_epoch,batch_idx)))
                right=len(seq)-self.random_crop_len
                left=0
                min_start=max(left,pos-self.random_crop_len+1)
                max_start=min(right,pos)
                if pos>=len(seq):start=len(seq)-self.random_crop_len
                else: start=np.random.randint(low=min_start,high=max_start+1)
                seq_after=seq[start:start+self.random_crop_len]
            else:
                seq_after=seq
                start=None
            seqs_after.append(seq_after)
            starts.append(start)
            positions.append(pos)
        return seqs_after,starts,positions


    def random_crop(self,seq,pos):
        np.random.seed(int('%d%d'%(self.trainer.current_epoch,self.trainer.global_step)))
        right=len(seq)-self.random_crop_len
        left=0
        min_start=max(left,pos-self.random_crop_len+1)
        max_start=min(right,pos)
        if pos>=len(seq):start=len(seq)-self.random_crop_len
        else: start=np.random.randint(low=min_start,high=max_start+1)
        seq_after=seq[start:start+self.random_crop_len]
        return seq_after
    
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
            seqs,starts=self.random_crop_batch(batch,batch_idx)
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
        
        loss=nn.functional.cross_entropy(y,batch_labels)
        torch.cuda.empty_cache()
        self.train_out.append(torch.hstack([y,batch_labels.reshape(batch_size,1)]).cpu())
        del y,batch_labels,sequence_representations,batch_tokens
        return loss
    
    def on_train_epoch_end(self) :
        all_preds=torch.vstack(self.train_out)
        all_preds_gather=self.all_gather(all_preds).view(-1,3)
        train_auroc=self.auroc(all_preds.float()[:,1],all_preds.long()[:,-1])
        train_auroc_gather=self.auroc(all_preds_gather.float()[:,1],all_preds_gather.long()[:,-1])
        train_loss=nn.functional.cross_entropy(all_preds[:,:-1],all_preds.long()[:,-1])
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
        val_loss=nn.functional.cross_entropy(all_preds[:,:-1],all_preds.long()[:,-1])
        self.log('val_auroc_gathered',val_auroc_gather)
        if self.trainer.global_rank==0:
            print('\n\n------gathered auroc is %s----\n\n'%val_auroc_gather)

        del all_preds, val_auroc,val_loss
        self.val_out.clear()

    def test_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        self.esm_model.eval()
        self.proj.eval()
        idxes,labels,seqs=batch['idx'],batch['label'].long(),batch['seq']
        batch_sample=list(zip(labels,seqs))
        del batch
        batch_labels, _, batch_tokens=self.batch_converter(batch_sample)
        batch_tokens=batch_tokens.to(self.device)
        batch_labels=torch.stack(batch_labels)
        batch_labels=batch_labels.to(self.device)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        sequence_representations=self.train_mul_gpu(batch_tokens)
        y=self.proj(sequence_representations.float().to(self.device))
        loss=nn.functional.cross_entropy(y,batch_labels)
        
        self.log('train_ce_loss',loss)
        torch.cuda.empty_cache()
        return loss

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
    def __init__(self, esm_model=esm.pretrained.esm2_t12_35M_UR50D(),esm_model_dim=480,n_class=2,truncation_len=None,unfreeze_n_layers=3,repr_layers=12,batch_sample='random',include_wild=False,lr=None,random_crop_len=None,debug=False):
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
        super().__init__(esm_model,esm_model_dim,truncation_len,unfreeze_n_layers,repr_layers,lr,random_crop_len=random_crop_len)
        self.batch_sample=batch_sample
        self.include_wild=include_wild
        self.init_dataset()
        self.auroc=BinaryAUROC()
        if self.include_wild:self.embs_dim=2*self.esm_model_dim
        else:self.embs_dim=self.esm_model_dim
        self.proj=nn.Sequential(
            nn.Linear(self.embs_dim,2),
            nn.Softmax(dim=1)
        )
        self.val_out=[]
        self.train_out=[]
        self.debug=debug
    def init_dataset(self):
        self.dataset=ProteinSequence()

    def print_memory(self,step,detail=False):
        if self.debug: 
            mem = cuda.memory_allocated() / 1024 ** 3  
            print('/n/n/n====== %s memory used:%.2f=============/n'%(step,mem))
            if detail and mem>10:
                for name, param in self.named_parameters():
                    if param.requires_grad:
                        param_memory = param.element_size() * param.numel() / 1024 ** 3  # Memory usage for the parameter
                        if param_memory>1:
                            print(f"{name} memory usage: {param_memory:.5f} GB")

                for name, buffer in self.named_buffers():
                    buffer_memory = buffer.element_size() * buffer.numel() / 1024 ** 3  # Memory usage for the buffer
                    if buffer_memory>1:
                        print(f"{name} memory usage: {buffer_memory:.5f} GB")

                for name, tensor in self.__dict__.items():
                    if torch.is_tensor(tensor) and tensor.is_cuda:
                        tensor_memory = tensor.element_size() * tensor.numel() / 1024 ** 3  # Memory usage for the tensor
                        if tensor_memory>1:print(f"{name} memory usage: {tensor_memory:.5f} GB")

                for name, tensor in self.named_buffers():
                    if tensor.grad is not None:
                        tensor_grad_memory = tensor.grad.element_size() * tensor.grad.numel() / 1024 ** 3  # Memory usage for the tensor gradient
                        if tensor_grad_memory>1:print(f"{name}.grad memory usage: {tensor_grad_memory:.5f} GB")

                for name, param in self.named_parameters():
                    if param.grad is not None:
                        param_grad_memory = param.grad.element_size() * param.grad.numel() / 1024 ** 3  # Memory usage for the parameter gradient
                        if param_grad_memory>1:print(f"{name}.grad memory usage: {param_grad_memory:.5f} GB")
        
        else:
            pass
    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        labels=batch['label'].long()
        
        if self.random_crop_len:
            seqs,starts,pos=self.random_crop_batch(batch,batch_idx)
        else:
            starts=None
            seqs=batch['seq']
        len_seqs=[len(seq) for seq in seqs]
        print('\n\n\n\n\n\n\n===================================\nlength of seqs in this batch(%s) is %s\n========================='%(batch_idx,len_seqs))
        mutated_batch_samples=list(zip(labels,seqs))

        del seqs
        wild_batch_samples=self.get_wild_batch(batch,starts=starts,pos=pos)
        self.print_memory('initiated batches',detail=True)

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

        if self.include_wild:embs=torch.hstack([delta_embs,wild_embs])

        else:embs=delta_embs

        #TODO attention

        y=self.proj(embs.float().to(self.device))
        self.print_memory('after projection',detail=True)

        del delta_embs,mutated_embs,wild_embs,embs

        loss=nn.functional.cross_entropy(y,labels)
        torch.cuda.empty_cache()
        self.train_out.append(torch.hstack([y,labels.reshape(batch_size,1)]).cpu())

        del y,labels
        self.print_memory('end of the batch',detail=True)

        return loss
    
    def on_train_epoch_end(self) :
        all_preds=torch.vstack(self.train_out)
        all_preds_gather=self.all_gather(all_preds).view(-1,3)
        train_auroc=self.auroc(all_preds.float()[:,1],all_preds.long()[:,-1])
        train_auroc_gather=self.auroc(all_preds_gather.float()[:,1],all_preds_gather.long()[:,-1])
        train_loss=nn.functional.cross_entropy(all_preds[:,:-1],all_preds.long()[:,-1])
        self.log('train_loss',train_loss,sync_dist=True)
        self.log('train_auroc_gathered',train_auroc_gather)
        if self.trainer.global_rank==0:
            print('\n\n------gathered auroc is %s----\n\n'%train_auroc_gather)
        del all_preds, train_auroc,train_loss
        self.train_out.clear()

    def validation_step(self, batch, batch_idx):
        self.esm_model.eval()
        self.proj.eval()
        torch.cuda.empty_cache()
        labels,seqs=batch['label'].long(),batch['seq']
        mutated_batch_samples=list(zip(labels,seqs))
        wild_batch_samples=self.get_wild_batch(batch)

        mutated_embs=self.get_esm_embedings(mutated_batch_samples)
        wild_embs=self.get_esm_embedings(wild_batch_samples)

        del seqs,mutated_batch_samples,wild_batch_samples,batch
        delta_embs=mutated_embs-wild_embs
        if self.include_wild:embs=torch.hstack([delta_embs,wild_embs])
        else:embs=delta_embs

        batch_size=wild_embs.shape[0]

        y=self.proj(embs.float().to(self.device))
        del delta_embs,mutated_embs,wild_embs,embs
    
        torch.cuda.empty_cache()
        pred=torch.hstack([y,labels.reshape(batch_size,1)]).cpu()
        self.val_out.append(pred)
        return pred

    def on_validation_epoch_end(self):
        all_preds=torch.vstack(self.val_out)
        all_preds_gather=self.all_gather(all_preds).view(-1,3)
        val_auroc=self.auroc(all_preds.float()[:,1],all_preds.long()[:,-1])
        val_auroc_gather=self.auroc(all_preds_gather.float()[:,1],all_preds_gather.long()[:,-1])
        val_loss=nn.functional.cross_entropy(all_preds[:,:-1],all_preds.long()[:,-1])
        self.log('val_loss',val_loss,sync_dist=True)
        self.log('val_auroc_gathered',val_auroc_gather)
        val_auroc_average=torch.mean(self.all_gather(val_auroc),dim=0)
        if self.trainer.global_rank==0:
            print('gathered auroc is %s'%val_auroc_gather)
        del all_preds, val_auroc,val_loss
        self.val_out.clear()
    
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
        seqs=[get_sequence_from_uniprot_id(uniprot) for uniprot in uniprots]
        batch_sample=[]
        for i,seq in enumerate(seqs):
            if starts is None:
                if len(seq)>self.random_crop_len: # if mutated seq is not cropped but the wild is too long
                    seq=self.random_crop(seq,pos[i])
                    batch_sample.append((uniprots[i],seq))
                else:
                    batch_sample.append((uniprots[i],seq))
            else:
                if starts[i]: #if mutated sequences are cropped and a start is returned
                    seq=seq[starts[i]:starts[i]+self.random_crop_len]
                    batch_sample.append((uniprots[i],seq))
                elif starts[i] is None and len(seq)>self.random_crop_len: # if mutated seq is not cropped but the wild is too long
                    seq=self.random_crop(seq,pos[i])
                    batch_sample.append((uniprots[i],seq))
                else:
                    batch_sample.append((uniprots[i],seq))

                
        print('/n/n length of wild batch is %s'%[len(seq) for seq in seqs])
        return batch_sample



        