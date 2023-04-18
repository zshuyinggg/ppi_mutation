import torch
from torch.utils.data import Dataset

global top_path  # the path of the top_level directory
global script_path, data_path, logging_path
import os, sys
from torch.utils.data import DataLoader
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
from scripts.esm.datasets import *
import pandas as pd
if __name__ == '__main__':
    dataset = ProteinSequence(os.path.join(script_path, 'merged_2019_1.csv'),
                           data_path + '/2019_1_sequences_terminated.csv', gen_file=False, all_uniprot_id_file= \
                               os.path.join(data_path, 'single_protein_seq/uniprotids_humap_huri.txt'), \
                           test_mode=False)

# model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
# batch_converter = alphabet.get_batch_converter(truncation_seq_length=20) #TODO: 1)what is a reasonable trancation length. 2) we do not want to trunca
# dataloader = DataLoader(Test, batch_size=2,
#                         shuffle=True, num_workers=1)
# model.eval()
#
# for i_batch, sample_batched in enumerate(dataloader):
#     # print(sample_batched)
#
#     sample_batched=list(zip(*sample_batched))
#
#     batch_labels, batch_strs, batch_tokens = batch_converter(sample_batched)
#     #batch=list(zip(*sample_batched))
#     batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
#     with torch.no_grad():
#         results = model(batch_tokens, repr_layers=[33], return_contacts=True)
#     token_representations = results["representations"][33]
#     sequence_representations = []
#     for i, tokens_len in enumerate(batch_lens):
#         sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))
#     print('i_batch %s'% i_batch)
# # %%
