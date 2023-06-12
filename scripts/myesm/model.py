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
    def __init__(self, esm_model=esm.pretrained.esm2_t36_3B_UR50D(),esm_model_dim=2560,truncation_len=None,unfreeze_n_layers=10,repr_layers=36):
        super().__init__()
        self.val_out=[]
        self.save_hyperparameters()
        self.esm_model_dim=esm_model_dim
        self.esm_model, alphabet=esm_model
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

    def freeze_layers(self):
        num=self.repr_layers-self.unfreeze_n_layers
        for layer in self.esm_model.named_parameters():
            if 'layers' in layer[0] and int(layer[0].split('.')[1])<num:
                layer[1].requires_grad=False

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        _,labels,seqs=batch['idx'],batch['label'].long(),batch['seq']
        batch_sample=list(zip(labels,seqs))
        del batch
        batch_labels, _, batch_tokens=self.batch_converter(batch_sample)
        batch_tokens=batch_tokens.to(self.device)
        # print(batch_tokens.shape)
        batch_size=batch_tokens.shape[0]

        batch_labels=torch.stack(batch_labels).reshape(batch_size)
        batch_labels=batch_labels.to(self.device)

        sequence_representations=self.train_mul_gpu(batch_tokens)
        y=self.proj(sequence_representations.float().to(self.device))
        
        loss=nn.functional.cross_entropy(y,batch_labels)
        torch.cuda.empty_cache()
        self.train_out.append(torch.hstack([y,batch_labels.reshape(batch_size,1)]).cpu())
        del y,batch_labels,sequence_representations,batch_tokens,batch_sample,labels,seqs
        return loss
    
    def on_train_epoch_end(self) :
        all_preds=torch.vstack(self.train_out)
        train_auroc=self.auroc(all_preds.float()[:,1],all_preds.long()[:,-1])
        train_loss=nn.functional.cross_entropy(all_preds[:,:-1],all_preds.long()[:,-1])
        self.log('train_loss',train_loss,sync_dist=True)
        self.log('train_auroc',train_auroc,sync_dist=True)
        del all_pred, train_auroc,train_loss
        self.train_out.clear()

    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        self.esm_model.eval()
        self.proj.eval()
        _,labels,seqs=batch['idx'],batch['label'].long(),batch['seq']
        batch_sample=list(zip(labels,seqs))
        del batch
        batch_labels, _, batch_tokens=self.batch_converter(batch_sample)
        batch_tokens=batch_tokens.to(self.device)
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
        val_auroc=self.auroc(all_preds.float()[:,1],all_preds.long()[:,-1])
        val_loss=nn.functional.cross_entropy(all_preds[:,:-1],all_preds.long()[:,-1])
        self.log('val_loss',val_loss,sync_dist=True)
        self.log('val_auroc',val_auroc,sync_dist=True)
        self.val_out.clear()
        del all_preds,val_auroc,val_loss

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

    def configure_optimizers(self,lr=4*1e-5) :
        optimizer=optim.Adam(self.parameters(),lr=lr)
        return optimizer

    def train_mul_gpu(self,batch_tokens):
        torch.cuda.empty_cache()
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
    def __init__(self, esm_model=esm.pretrained.esm2_t12_35M_UR50D(),esm_model_dim=480,n_class=3,truncation_len=None,unfreeze_n_layers=3,repr_layers=12,batch_sample='random',include_wild=False):
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
        super().__init__(esm_model,esm_model_dim,truncation_len,unfreeze_n_layers,repr_layers)
        self.batch_sample=batch_sample
        self.include_wild=include_wild
        self.init_dataset()
        self.auroc=BinaryAUROC()
        if self.include_wild:self.embs_dim=2*self.esm_model_dim
        else:self.embs_dim=self.esm_model_dim
        self.proj=nn.Sequential(
            nn.Linear(self.embs_dim,2),
            nn.Softmax()
        )
        self.val_out=[]
        self.train_out=[]
    def init_dataset(self):
        self.dataset=ProteinSequence()


    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        labels,seqs=batch['label'].long(),batch['seq']
        mutated_batch_samples=list(zip(labels,seqs))

        del seqs
        wild_batch_samples=self.get_wild_batch(batch)

        try:
            mutated_embs=self.get_esm_embedings(mutated_batch_samples)
            wild_embs=self.get_esm_embedings(wild_batch_samples)
        except AttributeError:
            print(seqs)
            return 0
        batch_size=wild_embs.shape[0]

        del mutated_batch_samples,wild_batch_samples,batch
        #TODO: multichannel mlp
        delta_embs=mutated_embs-wild_embs
        if self.include_wild:embs=torch.hstack([delta_embs,wild_embs])
        else:embs=delta_embs

        #TODO attention

        y=self.proj(embs.float().to(self.device))
        del delta_embs,mutated_embs,wild_embs,embs

        loss=nn.functional.cross_entropy(y,labels)
        # self.log('train_loss',loss,on_epoch=True,on_step=False,sync_dist=True) #TODO? average epoch?
        torch.cuda.empty_cache()
        self.train_out.append(torch.hstack([y,labels.reshape(batch_size,1)]).cpu())
        # train_auroc=self.auroc(y,batch_labels)
        # self.log('train_auroc',train_auroc,on_epoch=True, sync_dist=True)
        del y,labels
        return loss
    
    def on_train_epoch_end(self) :
        all_preds=torch.vstack(self.train_out)
        # print(all_preds.shape)
        train_auroc=self.auroc(all_preds.float()[:,1],all_preds.long()[:,-1])
        train_loss=nn.functional.cross_entropy(all_preds[:,:-1],all_preds.long()[:,-1])
        self.log('train_loss',train_loss)
        self.log('train_auroc',train_auroc)
        self.train_out.clear()

    def validataion_step(self, batch, batch_idx):
        self.esm_model.eval()
        self.proj.eval()
        torch.cuda.empty_cache()
        labels,seqs=batch['label'].long(),batch['seq']
        mutated_batch_samples=list(zip(labels,seqs))
        wild_batch_samples=self.get_wild_batch(batch)

        mutated_embs=self.get_esm_embedings(mutated_batch_samples)
        wild_embs=self.get_esm_embedings(wild_batch_samples)

        del seqs,mutated_batch_samples,wild_batch_samples,batch
        #TODO: multichannel mlp
        delta_embs=mutated_embs-wild_embs
        if self.include_wild:embs=torch.hstack([delta_embs,wild_embs])
        else:embs=delta_embs

        #TODO attention
        batch_size=wild_embs.shape[0]

        y=self.proj(embs.float().to(self.device))
        del delta_embs,mutated_embds,wild_embs,embs
        # loss=nn.functional.cross_entropy(y,labels)
        # self.log('val_loss',loss,prog_bar=True, on_step=False, on_epoch=True)
        # torch.cuda.empty_cache()
        # val_auroc=self.auroc(y,labels)
        # self.log('val_auroc',val_auroc,prog_bar=True, on_step=False, on_epoch=True)
        torch.cuda.empty_cache()
        print(y.shape,labels.shape)
        pred=torch.hstack([y,labels.reshape(batch_size,1)].cpu())
        self.val_out.append(pred)
        return pred

    def on_validation_epoch_end(self):
        all_preds=torch.vstack(self.val_out)
        # print(all_preds.shape)
        val_auroc=self.auroc(all_preds.float()[:,1],all_preds[:,-1])
        val_loss=nn.functional.cross_entropy(all_preds[:,:-1],all_preds.long()[:,-1])
        self.log('val_loss',val_loss,sync_dist=True)
        self.log('val_auroc',val_auroc,sync_dist=True)
        self.val_out.clear()

    
    def get_esm_embedings(self,batch_sample):
        torch.cuda.empty_cache()
        _, _, batch_tokens=self.batch_converter(batch_sample)
        batch_tokens=batch_tokens.to(self.device)
        if batch_tokens.shape[1]>512:batch_tokens=batch_tokens[:,:512] #TODO random crop
        # batch_labels=torch.stack(batch_labels)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        results = self.esm_model(batch_tokens, repr_layers=[self.repr_layers], return_contacts=False)
        token_representations = results["representations"][self.repr_layers]
        del results
        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))
        del token_representations
        torch.cuda.empty_cache()
        sequence_representations=torch.stack(sequence_representations)
        del batch_lens,batch_tokens
        return sequence_representations

    def get_wild_batch(self,mutated_batch):
        uniprots=mutated_batch['UniProt']
        batch_sample=[(uniprot,get_sequence_from_uniprot_id(uniprot)) for uniprot in uniprots]
        return batch_sample



        