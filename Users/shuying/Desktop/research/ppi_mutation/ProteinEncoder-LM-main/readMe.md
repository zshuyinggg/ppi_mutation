# Setup Environment
Please check steps in __setup_env.sh__ to set up the environment. Some variables need to changed according to user's preference.

# Protein Sequence-based Language Modeling

## Compute Residue Embeddings for Protein Sequences
1. Prepare input data

For each sequence, save the information of identifier and amino acids in the dictionary format, follow the example below. And append all such dictionaries to a List (e.g. named data_list).
```text
{'seq_id': 'protein_seq_1' # change to your identifier, unique for each sequence
 'seq_primary': 'VQLVQSGAAVKKPGESLRISCKGSGYIFTNYWINWVRQMPGRGLEWMGRIDPSDSYTNYSSSFQGHVTISADKSISTVYLQWRSLKDTDTAMYYCARLGSTA' # string, upper cases
 }
```

Then, use the following code to save into lmdb format.
```python
import lmdb
import pickle as pkl

map_size = (1024 * 15) * (2 ** 20) # 15G, change accordingly
wrtEnv = lmdb.open('path/to/data/dir/data_file_name.lmdb',map_size=map_size)
with wrtEnv.begin(write=True) as txn:
    for i, entry in enumerate(data_list): # data_list contains all dictionaries in the above format
        txn.put(str(i).encode(), pkl.dumps(entry))
    txn.put(b'num_examples', pkl.dumps(i+1))
wrtEnv.close()
```

2. download trained models from [google drive](https://drive.google.com/file/d/1FZewUpVQ2jJL_Hg5NyFM6qGb4exJ2SRr/view?usp=sharing), decompress and save under the main folder ProteinEncoder-LM/

3. Run model

Run the following command to generate embeddings. If you use any relative path inputs for some parameters, make sure it is visible under the 'ProteinEncoder-LM/' folder. Please change parameters in '{}' accordingly (delete '{}' and comments afterwards)

```python
python scripts/main.py \
    run_eval \
    transformer \
    embed_seq \
    {'/path/to/saved_model'}  (e.g. 'trained_models/rp15_pretrain_1_models') \
    --batch_size {4} (change accordingly) \
    --data_dir {'/path/to/dataset'} \
    --metrics save_embedding \
    --split {'data_file_name'} \
    --embed_modelNm {'customized identifier for model'} (e.g. 'rp15_pretrain_1')
```
For a sequence of length L, the final embeddings have size L * 768. After successful running of the python command, a json file 'embedding_{data_file_name}_{embed_modelNm}.json' should appear under the folder '/path/to/dataset'. Sequence identifiers and embeddings are organized in the dictionary format below.
```text
{seq_id : embedding_matrix} // seq_id is the given identifier for the sequence, embedding_matrix is a list with size L * 768.
```

## Compute residue embeddings for antibody heavy and light chain sequences
1. Prepare input data

For each antibody heavy and light chain pair, the sequence data needs to be saved into this dictionary format with unique identifiers for H/L sequence which will be used as identifiers for embedding later.
```txt
{"entityH": '001_VH', //unique identifier for heavy chain sequence
 "entityL": '001_VL', //unique identifier for light chain sequence
 "seqVH": 'EVQLVQSGAAVKKPGESLRISCKGSGYIFTNYWINWVRQMPGRGLEWMGRIDPSDSYTNYSSSFQGHVTISADKSISTVYLQWRSLKDTDTAMYYCARLGSTAPWGQGTMVTVSS', //VH sequence
 "seqVL": 'DIQMTQSPSSLSASVGDRLTITCRASQSIDNYLNWYQQKPGKAPQLLIYGASRLQDGVSSRFSGSGSGTDFTLTISSLQPEDFATYFCQQGYSVPFTFGPGTKLDIK', //VL sequence
}
```
Then, use this code to save into lmdb format.
```python
import lmdb
import pickle as pkl

map_size = (1024 * 15) * (2 ** 20) # 15G, change accordingly
wrtEnv = lmdb.open('path/to/data/dir/data_file_name.lmdb',map_size=map_size)
with wrtEnv.begin(write=True) as txn:
    for i, entry in enumerate(data_list): # data_list contains all dictionaries in the above format
        txn.put(str(i).encode(), pkl.dumps(entry))
    txn.put(b'num_examples', pkl.dumps(i+1))
wrtEnv.close()
```
2. Run model

Download the trained model's weight and configuration files from [here](https://drive.google.com/drive/folders/1vuMRUwAqX0iIuJ0EfqbgT0ppDpWdFk4G?usp=sharing) and save under a local folder. For parameter *task*, refer to the column 'task' in [table](https://docs.google.com/document/d/1eGh1QT6j3FpSMPu8Sgfm5HBcABGMpraI_aI2HmMJ3Uc/edit?usp=sharing). Run python script below to compute embeddings for each residue.
```python
python scripts/main.py \
    run_eval \
    transformer \ 
    'task' \ # refer to the table
    'path/to/model/dir' \ # path to downloaded model dir
    --batch_size=16 \ # change accordingly
    --data_dir='path/to/data/dir' \ # path to data dir
    --split='data_file_name' \ # data file name without extension '.lmdb'
    --metrics embed_antibody
```
For VH of length L_h and VL of length L_l, the final embeddings have size L_h * hidden_dim for VH and L_l * hidden_dim for VL. After successful running of the command, a json file 'data_file_name.json' should appear under the folder 'path/to/data/dir/embeddings/'. Identifiers and embeddings are organized in the format below.
```text
{"entityH":  '001_VH', //unique identifier for heavy chain sequence
"entityL": '001_VL', //unique identifier for light chain sequence
"hidden_states_lastLayer_token_VH": list, // VH embeddings L_h * hidden_dim
"hidden_states_lastLayer_token_VL": list, // VL embeddings L_l * hidden_dim
}
```

## Finetuning over sequence data

## Predict mutation effect scores (ratio of likelihood)
* Download mutant set data and model weights from [shared drive](https://drive.google.com/drive/folders/1jeSUCLB9h4k37rgsLWTcL4yYBuZkgmhe?usp=sharing) then uncompress. Save under the project folder

* Predict mutation effect with code below. The results will be saved in a folder named 'mutagenesis_fitness_unSV' under model folder

```
python scripts/scripts_HfaceApi/main.py \
    run_eval \
    transformer \
    mutation_fitness_UNsupervise_mutagenesis \
    absolute/path/to/model/folder (e.g. ~/models/AMIE_PSEAE/bert_rp15_1) \
    --batch_size 16 \
    --data_dir absolute/path/to/root/folder/of/mutant/set (e.g. ~/mut_scan_data) \
    --metrics fitness_unsupervise_mutagenesis \
    --mutgsis_set mutant_set_name (e.g. AMIE_PSEAE_Whitehead)
```
