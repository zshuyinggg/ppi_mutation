import copy
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

import torch
from torch.utils.data import Dataset
import torch.utils.data as data

global top_path  # the path of the top_level directory
global script_path, data_path, logging_path
import os, sys
from torch.utils.data import DataLoader
import esm
from dask.diagnostics import ProgressBar

import lightning.pytorch as pl

from sklearn.model_selection import train_test_split


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
from scripts.utils_clinvar import *

from scripts.utils import *
import pandas as pd
import dask.dataframe as ddf
import multiprocessing
from torchvision import transforms, utils

# num_partitions = multiprocessing.cpu_count()-4
num_partitions = 28


def all_uniprot(f_path=pj(top_path, 'ppi_seq_huri_humap.csv')):
    uniprots, seqs = pd.read_csv(f_path)['UniProt'].unique().tolist(), pd.read_csv(f_path)['Seq'].tolist()
    return uniprots, seqs


class ProteinEmbeddings(Dataset):
    def __init__(self, embedding_path, clinvar_csv) -> None:
        super().__init__()
        self.embeddings = torch.load(embedding_path)
        self.all_sequences = pd.read_csv(clinvar_csv)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx, uniprot=None, label=None):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sequences = self.all_sequences.iloc[idx, self.all_sequences.columns.get_loc('Seq')]
        uniprot = self.all_sequences.iloc[idx, self.all_sequences.columns.get_loc('UniProt')]
        name = self.all_sequences.iloc[idx, self.all_sequences.columns.get_loc('Name')]
        labels = self.all_sequences.iloc[idx, self.all_sequences.columns.get_loc('Label')]
        sample = {'label': torch.tensor(labels).int(), 'UniProt': uniprot, 'Name': name,
                  'Loc': get_loc_from_name(name)}  # multiple or single?

        return sample


class ProteinSequence(Dataset):
    """
    Generate protein sequences Dataset
    """

    def __init__(self, clinvar_csv=os.path.join(script_path, 'merged_2019_1.csv'),
                 transform=None,
                 random_seed=52,
                 variant_only=True, wild_uniprots_list_2_include=[]):
        """
        labels: 2 wild. 0 negative. 1 possitive
        :param clinvar_csv:
        :param train_val_ratio: ration for training set
        :param random_seed:
        """
        self.random_seed = random_seed
        print('Reading file')
        self.all_sequences = pd.read_csv(clinvar_csv)
        self.transform = transform
        self.remove_na()
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        self.all_sequences = self.all_sequences.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        # set this to class property to make sure train and val are split on the same indexes
        self.all_sequences.loc[
            self.all_sequences['Name'] == '0', 'Label'] = 0
        self.all_sequences.loc[self.all_sequences['Label'] == -1, 'Label'] = 0
        print('!!!!========= \n Initial data to use: \n=========================',
              self.all_sequences['Label'].value_counts())

        if variant_only:
            self.all_sequences = self.all_sequences[self.all_sequences['Name'] != '0']
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
            self.all_sequences = self.all_sequences.sample(frac=1, random_state=random_seed).reset_index(drop=True)
            print('!!!!====Wild type is not included!!!! ========= \n Final data to use: \n',
                  self.all_sequences['Label'].value_counts())

        else:  # include all wild from ppi
            self.all_wild_uniprots, all_wild_seqs = all_uniprot()
            self.wild_df = pd.DataFrame(columns=['idx', 'Seq', 'label', 'UniProt', 'Name'])
            self.wild_df['UniProt'] = self.all_wild_uniprots
            self.wild_df['Name'] = 'wild_' + self.wild_df['UniProt'] + 'p.%d' % (
                    len(self.wild_df['Seq']) // 2)
            self.wild_df['Label'] = 0
            self.all_sequences = pd.concat([self.all_sequences, self.wild_df])
            self.all_sequences = self.all_sequences.sample(frac=1, random_state=random_seed).reset_index(drop=True)
            print('!!!!====Wild type is included!!!! ========= \n Final data to use: \n',
                  self.all_sequences['Label'].value_counts())

    def get_idx_from_uniprot(self, uniprot):
        idx = self.all_sequences[self.all_sequences['UniProt'] == uniprot].index
        return idx

    def remove_na(self):
        self.all_sequences.dropna(inplace=True, ignore_index=True)
        print('nan removed')

    def set_class_seq(self):
        ProteinSequence.all_sequences = self.all_sequences

    def correct_labels(self):

        # self.all_sequences['Label']=(self.all_sequences['Label']+1)/2
        self.all_sequences.loc[self.all_sequences['Name'] == '0', 'Label'] = 2
        print(self.all_sequences['Label'].describe())
        # self.all_sequences.to_csv(self.gen_file_path)

    def __len__(self):
        return len(self.all_sequences)

    def __getitem__(self, idx, uniprot=None, label=None):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sequences = self.all_sequences.iloc[idx, self.all_sequences.columns.get_loc('Seq')]
        uniprot = self.all_sequences.iloc[idx, self.all_sequences.columns.get_loc('UniProt')]
        name = self.all_sequences.iloc[idx, self.all_sequences.columns.get_loc('Name')]
        labels = self.all_sequences.iloc[idx, self.all_sequences.columns.get_loc('Label')]
        sample = {'idx': torch.tensor(idx).float(), 'Seq': sequences, 'label': torch.tensor(labels).int(),
                  'UniProt': uniprot, 'Name': name,
                  'Loc': get_loc_from_name(name)}  # multiple or single?
        if self.transform:
            sample = self.transform(sample)
        return sample

    def shuffle(self):
        print('dataset shuffled')
        return self.all_sequences.sample(frac=1, random_state=self.random_seed)

    def sort(self, ascending=True):
        s = self.all_sequences.Seq.str.len().sort_values(ascending=ascending).index
        df = self.all_sequences.reindex(s)
        self.all_sequences = df.reset_index(drop=True)

        return df


class ProteinWildSequence(Dataset):
    def __init__(self, file_path='/scratch/user/zshuying/ppi_mutation/ppi_seq_huri_humap.csv'):
        self.df_ppi = pd.read_csv(file_path)

    def __len__(self):
        return len(self.df_ppi)

    def __getitem__(self, idx):
        uniprot, sequence = self.df_ppi.iloc[idx, :]['UniProt'], self.df_ppi.iloc[idx, :]['Seq']
        return {'Seq': sequence, 'UniProt': uniprot}


def read_name_list(list_name):
    pth = pj('/scratch/user/zshuying/ppi_mutation/data/baseline1/processed', '%s.txt' % list_name)
    with open(pth, 'r') as f:
        name_list = eval(f.readline())
    return name_list


def get_subset_clinvar_by_name_list(df, list_name):
    name_list = read_name_list(list_name)
    subdf = df[df['Name'].apply(lambda x: x in name_list)]
    return subdf


class ProteinDataModule(pl.LightningDataModule):
    def __init__(self, logger, train_val_ratio=0.9,
                 clinvar_csv=os.path.join(script_path, 'merged_2019_1.csv'), variant_only=False, random_seed=1050,
                 num_workers=8,
                 read_variant_from_list_file=False,
                 variant_train_list_name=None,
                 variant_val_list_name=None,
                 read_variant_n_wild_from_list_file=False,
                 variant_n_wild_train_list_name=None,
                 variant_n_wild_val_list_name=None,
                 batch_size=4, write_list=False, write_train_name_list_name='2019_train_name_list',
                 write_val_name_list_name='2019_val_name_list', **args):
        super().__init__()
        self.dataset = ProteinSequence(clinvar_csv=clinvar_csv, variant_only=variant_only, random_seed=random_seed)
        self.logger = logger
        self.seed = random_seed

        if read_variant_n_wild_from_list_file:
            assert variant_only == False, "you set 'read_variant_n_wild_from_list_file' to True but 'variant_only' True too. which do you want?"
            train_set = get_subset_clinvar_by_name_list(self.dataset.all_sequences, variant_n_wild_train_list_name)
            val_set = get_subset_clinvar_by_name_list(self.dataset.all_sequences, variant_n_wild_val_list_name)

            self.logger.experiment.add_text('Data and Model Setups',
                                            'training set is read from the list %s\n validation set is read from the list %s\n' % (
                                                variant_n_wild_train_list_name, variant_n_wild_val_list_name),
                                            global_step=0)
        elif read_variant_from_list_file:
            variant_train_list = read_name_list(variant_train_list_name)
            variant_val_list = read_name_list(variant_val_list_name)
            if variant_only:
                train_set = get_subset_clinvar_by_name_list(self.dataset.all_sequences, variant_train_list)
                val_set = get_subset_clinvar_by_name_list(self.dataset.all_sequences, variant_val_list)
                self.log_set_up('read variant from\n %s and %s\nNOT stacking wild sequences' % (
                    variant_train_list_name, variant_val_list_name))

            else:
                wild_train_list, wild_val_list = self.gen_wild_train_val_list(train_val_ratio)
                train_set = get_subset_clinvar_by_name_list(self.dataset.all_sequences,
                                                            variant_train_list + wild_train_list)
                val_set = get_subset_clinvar_by_name_list(self.dataset.all_sequences,
                                                          variant_val_list + wild_val_list)
                self.log_set_up('read variant from\n %s and %s\nstack wild sequences to the data with ratio %s' % (
                    variant_train_list_name, variant_val_list_name, train_val_ratio))

        else:
            self.log_set_up(
                'Splitting train val with ratio = %s, did not seperate training set with lengths' % train_val_ratio)
            train_set, val_set = split_train_val(self.dataset, train_val_ratio, random_seed=self.seed)

        self.my_train_dataloader = DataLoader(train_set, batch_size=batch_size,
                                              shuffle=False, num_workers=num_workers, drop_last=True)
        self.my_val_dataloader = DataLoader(val_set, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers, drop_last=True)

        if write_list:
            train_name_list = train_set.dataset.all_sequences.iloc[train_set.indices]['Name'].tolist()
            val_name_list = val_set.dataset.all_sequences.iloc[val_set.indices]['Name'].tolist()
            with open(pj(
                    '/scratch/user/zshuying/ppi_mutation/data/baseline1/processed/%s_%s' % (
                            write_train_name_list_name, self.seed)), 'w') as f:
                f.writelines(str(train_name_list))
            with open(pj(
                    '/scratch/user/zshuying/ppi_mutation/data/baseline1/processed/%s_%s' % (
                            write_val_name_list_name, self.seed)),
                    'w') as f:
                f.writelines(str(val_name_list))

    def log_set_up(self, text):
        self.logger.experiment.add_text('Data and Model Setups', text, global_step=0)
        print(text, flush=True)

    def gen_wild_train_val_list(self, train_val_ratio):
        wild_list = self.dataset.all_wild_uniprots
        wild_train_list, wild_val_list = train_test_split(wild_list, train_val_ratio)
        return wild_train_list, wild_val_list

    def train_dataloader(self):
        return self.my_train_dataloader

    def val_dataloader(self):
        return self.my_val_dataloader

    def test_dataloader(self):
        return self.my_val_dataloader


class AllProteinVariantData(ProteinDataModule):
    """
    Subclass of ProteinSequence, without dividing train,val,test
    """

    def __init__(self, clinvar_csv=os.path.join(script_path, 'merged_2019_1.csv'), batch_size=20, num_workers=15):
        super().__init__(clinvar_csv=clinvar_csv)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = ProteinSequence(clinvar_csv=clinvar_csv)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          drop_last=True)


class AllWildData(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.dataset = ProteinWildSequence()

    def train_dataloader(self):
        return DataLoader(self.dataset, shuffle=False, batch_size=20, num_workers=15)

    def val_dataloader(self):
        return DataLoader(self.dataset, shuffle=False, batch_size=20, num_workers=15)


class EsmMeanEmbeddings(Dataset):
    def __init__(self, if_initial_merge=False, dirpath=data_path):
        self.if_initial_merge = if_initial_merge
        self.dirpath = dirpath
        if self.if_initial_merge: self.initial_merge()
        self.read_file()

    def initial_merge(self):
        pattern = r'all.*predictions_[0-9]+.*'
        # Compile the regular expression
        regex = re.compile(pattern)
        # Loop over all files in the directory
        preds = []
        for file in os.listdir(self.dirpath):
            if os.path.isfile(os.path.join(self.dirpath, file)):
                print(file)
                # Check if the file matches the pattern
                if regex.search(file):
                    pred = torch.load(os.path.join(self.dirpath, file))
                    preds.append(pred)
                    del pred
                    print('file %s has been loaded ' % file)
        preds = torch.vstack(preds)
        print('predictions merged')
        torch.save(preds, os.path.join(data_path, '2019_all_esm_embeddings.pt'))
        print('predictions saved')
        pattern = 'all.*_labels_[0-9]+.*'
        # Compile the regular expression
        regex = re.compile(pattern)
        # Loop over all files in the directory
        preds = []
        for file in os.listdir(self.dirpath):
            if os.path.isfile(os.path.join(self.dirpath, file)):
                # Check if the file matches the pattern
                if regex.search(file):
                    pred = torch.load(os.path.join(self.dirpath, file))
                    preds.append(pred)
                    del pred
                    print('file %s has been loaded ' % file)

        preds = np.concatenate(preds, axis=0)
        print('labels merged')
        torch.save(preds, os.path.join(data_path, '2019_all_labels_for_embeddings.pt'))
        print('labels saved')

    def read_file(self):

        self.embeddings = torch.load(os.path.join(data_path, '2019_all_esm_embeddings.pt'))
        self.labels = torch.load(os.path.join(data_path, '2019_all_labels_for_embeddings.pt'))
        print(self.embeddings.shape)
        print(self.labels.shape)
        self.labels = (self.labels + 1) / 2

    def __len__(self):
        assert len(self.labels) == len(self.embeddings)
        return len(self.labels)

    def __getitem__(self, idx):
        return {'embedding': self.embeddings[idx, :],
                'label': self.labels[idx]}


def split_train_val(dataset, train_val_split=0.8, random_seed=52):
    train_set_size = int(len(dataset) * train_val_split)
    valid_set_size = len(dataset) - train_set_size
    seed = torch.Generator().manual_seed(random_seed)
    train_set, valid_set = data.random_split(dataset, [train_set_size, valid_set_size], generator=seed)
    print('Split dataset into train, val with the rate of %s' % train_val_split)
    print(train_set.dataset.all_sequences['Label'].value_counts())
    return train_set, valid_set
