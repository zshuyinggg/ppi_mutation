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
import pandas as pd


class ProteinSequence(Dataset):
    """
    Generate protein sequences according to provided clinvar_csv to gen_dir,
    label decides only positive samples(1), only negative samples(0), or both (None)
    """

    def __init__(self, clinvar_csv, gen_file_path, gen_file=True,
                 all_uniprot_id_file='data/single_protein_seq/uniprotids_humap_huri.txt',
                 test_mode=False):
        print('Reading file')
        self.clinvar = pd.read_csv(clinvar_csv)
        self.test_mode=test_mode
        self.gen_file_path = gen_file_path
        self.all_ppi_uniprot_ids = eval(open(all_uniprot_id_file).readline())
        self.clinvar = self.clinvar[
            [uniprot in self.all_ppi_uniprot_ids for uniprot in self.clinvar['UniProt'].tolist()]]
        # check if sequence file already exist:
        if test_mode:
            self.clinvar=self.clinvar.loc[:1000,:]
        if gen_file:
            self.gen_sequence_file()
        else:
            if os.path.isfile(gen_file_path):
                self.all_sequences = pd.read_csv(gen_file_path)
            else:
                self.gen_sequence_file()

    def gen_sequence_file(self) -> object:
        if self.test_mode:print('-----Test mode on-----------')
        print('Initiating datasets....\n')
        print('Generating mutant sequences...\n')
        # df_sequence_mutant = self.clinvar.loc[:, ['#AlleleID', 'label', 'UniProt', 'Name']]  # TODO review status
        # df_sequence_mutant['Seq'] = [gen_mutant_one_row(uniprot_id, name) for uniprot_id, name in \
        #                              zip(df_sequence_mutant['UniProt'], df_sequence_mutant['Name'])]
        len_wild = len(self.all_ppi_uniprot_ids)
        # df_sequence_mutant.to_csv(self.gen_file_path)
        df_sequence_mutant=pd.read_csv(self.gen_file_path)
        #TODO exclude those who is not in ppi

        print('Generating wild sequences...\n')
        self.all_ppi_uniprot_ids=list(self.all_ppi_uniprot_ids)
        if self.test_mode:
            self.all_ppi_uniprot_ids=self.all_ppi_uniprot_ids[:100]
            len_wild = 100

        df_sequence_wild = pd.DataFrame(0, index=np.arange(len_wild),
                                        columns=['#AlleleID', 'Label', 'UniProt', 'Name', 'Seq'])
        df_sequence_wild['UniProt'] = list(self.all_ppi_uniprot_ids)
        df_sequence_wild['Seq'] = [get_sequence_from_uniprot_id(id) for id in df_sequence_wild['UniProt']]
        df_sequence_wild['Label'] = [-1] * len_wild
        df_sequences = pd.concat([df_sequence_wild, df_sequence_mutant])
        df_sequences.to_csv(self.gen_file_path)
        self.all_sequences = df_sequences # TODO
        return df_sequences
        # self.wild_uniprotIDs = self.all_ppi_uniprot_ids
        # self.positive_data = self.clinvar[self.clinvar['label'] == 1]
        # self.positive_uniprotIDs = get_uniprot(self.positive_data['UniProt'])
        # self.positive_names = self.positive_data['Name']
        # self.negative_data = self.clinvar[self.clinvar['label'] == -1]
        # self.negative_uniprotIDs = get_uniprot(self.negative_data['UniProt'])
        # self.negative_names = self.negative_data['Name']
        # self.all_names = self.wild_uniprotIDs + self.positive_names.tolist() + self.negative_names.tolist()
        # self.all_IDs = self.wild_uniprotIDs + self.positive_data['#AlleleID'].tolist() + self.negative_data['#AlleleID']
        # self.positive_sequences = gen_mutants(self.positive_data, self.gen_dir + 'positive', bs=None, delimiter=',')
        # self.negative_sequences = gen_mutants(self.negative_data, self.gen_dir + 'negative', bs=None, delimiter=',')
        # self.wild_sequences = gen_sequences_oneFile(self.all_ppi_uniprot_ids, self.gen_dir + 'wild')
        # self.all_sequences = self.wild_sequences.update(self.positive_sequences).update(self.negative_sequences)
        # # return self.positive_sequences,self.negative_sequences,self.wild_sequences

    def __len__(self):
        return len(self.all_sequences)

    def __getitem__(self, idx, uniprot=None, label=None):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sequences = self.all_sequences.loc[idx, 'Seq']  # TODO check index
        # labels=self.all_sequences.loc[idx,'Label']
        return idx, sequences.replace('*','')


#
# class ESMProteinSeq(ProteinSequence):
#     def __int__(self):
#         super.__init__()
#
#     def __getitem__(self, idx, uniprot=None, label=None):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         if not uniprot:
#             if label == 1:  # all positive #TODO
#                 sequences_names = self.positive_names[idx]
#                 sequences = self.positive_sequences[sequences_names]
#             elif label == -1:  # all negative #TODO
#                 sequences_names = self.negative_names[idx]
#                 sequences = self.negative_sequences[sequences_names]
#             elif label == 0:  # all wildtype #TODO
#                 sequences_uniprotIDs = self.wild_uniprotIDs[idx]
#                 sequences = self.wild_sequences[sequences_uniprotIDs]
#             elif label is None:
#                 sequences_IDs = self.all_IDs[idx]
#                 sequences = [self.all_sequences[k] for k in sequences_IDs]
#         return self.sequences_IDs, sequences

# %%


# check_if_in_ppi('sss')sss
Test = ProteinSequence(os.path.join(script_path, 'merged_2019_1.csv'),
                       data_path + '/2019_1_test_sequences.csv', gen_file=False, all_uniprot_id_file= \
                           os.path.join(data_path, 'single_protein_seq/uniprotids_humap_huri.txt'),\
                       test_mode=True)
# All=ProteinSequence(os.path.join(script_path, 'merged_2019_1.csv'),
#                          data_path + '/2019_1_all_sequences.csv', gen_file=True, all_uniprot_id_file= \
#                              os.path.join(data_path, 'single_protein_seq/uniprotids_humap_huri.txt'), \
#                          test_mode=False)
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter(truncation_seq_length=2000) #TODO: 1)what is a reasonable trancation length. 2) we do not want to trunca
dataloader = DataLoader(Test, batch_size=4,
                        shuffle=True, num_workers=1)
model.eval()
for i_batch, sample_batched in enumerate(dataloader):
    # print(sample_batched)
    print('starting batch with length %s '%batch_lens)

    sample_batched=list(zip(*sample_batched))

    batch_labels, batch_strs, batch_tokens = batch_converter(sample_batched)
    #batch=list(zip(*sample_batched))
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))
    print('batch_lens %s succeeded'%batch_lens)
# %%
