

```python
%load_ext autoreload
```


```python
import os
import time
from fastai.text import *
```


```python
import sys
sys.path.append('./imdb_scripts/')

```


```python
from create_toks import *
import sentencepiece as spm
from fastai.text import *
```

### define common variables such as path, file prefixes... 


```python
currvocab = 8
is_lowercase = True
curr_vocab_str = f"{currvocab}k"
if is_lowercase: 
    curr_vocab_str = curr_vocab_str + '_lowercase'
```


```python
train_dir_str = 'data/aclImdb/train/all/'
all_name = 'train_all_lower.txt'
```


```python
BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag
SPM_MODEL_PATH=Path(f'data/aclImdb_spm/{curr_vocab_str}/')
PATH=Path('data/aclImdb/')
```


```python
CLAS_PATH=Path('data/imdb_clas/')
LM_PATH=Path('data/imdb_lm/')
CLAS_PATH_SPM=Path('data/imdb_clas_spm/')
LM_PATH_SPM=Path('data/imdb_lm_spm/')
```


```python
chunksize=24000
```

## First of all, all training and testing datasets are lowercased. 
This step is ommited for its simplicity. (There's literally 1000 ways to do it!)


```python
full_p = train_dir_str + all_name
to_txt = 'train_all_lower.txt'
```


```python
arg_str = '--input='+ train_dir_str + to_txt + f' --model_prefix=model_' + curr_vocab_str + f' --vocab_size={currvocab}000'
print(arg_str)
```

    --input=data/aclImdb/train/all/train_all_lower.txt --model_prefix=model_8k_lowercase --vocab_size=8000


### This is where google SentencePiece tokenizes the texts. (<5 min)
It can be called directly from commandline or inside python. 


```python
spm.SentencePieceTrainer.Train(arg_str)
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-12-e295cb6211df> in <module>()
    ----> 1 spm.SentencePieceTrainer.Train(arg_str)
    

    KeyboardInterrupt: 


## modify get_texts and get_all for SentencePiece, to turn dataframe into tokens
### google already threads it for you, so one less thing to worry about!


```python
def get_texts_spm(spm_model, df, n_lbls):
    tstart = time.time()
    if len(df.columns) == 1:
        labels = []
        texts = f'\n{BOS} {FLD} 1 ' + df[0].astype(str)
        texts = texts.apply(fixup).values.astype(str)
    else:
        labels = df.iloc[:,range(n_lbls)].values.astype(np.int64)
        texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
        for i in range(n_lbls+1, len(df.columns)): texts += f' {FLD} {i-n_lbls} ' + df[i].astype(str)
        texts = texts.apply(fixup).values.astype(str)

    #tok = proc_all_mp_smp(partition_by_cores(texts))
    tok = [spm_model.EncodeAsIds(t) for t in texts]
    tend = time.time()
    print(f'{(tend-tstart)/(len(texts)/1000):.2f}sec per 1k rows')
    return tok, list(labels)
```


```python
def get_all_spm(spm_model, df, n_lbls):
    tok, labels = [], []
    for i, r in enumerate(df):
        
        print(i)
        tok_, labels_ = get_texts_spm(spm_model, r, n_lbls)
        tok += tok_;
        labels += labels_
        
    return tok, labels
```


```python
sp8_lower = spm.SentencePieceProcessor()
```


```python
sp8_lower.Load('data/imdb_lm_spm/model_8k_lowercase.model')
```




    True




```python
print(sp8_lower.EncodeAsPieces("I wish I knew what the FUCKingHELL is up with the thingy".lower()))
```

    ['▁i', '▁wish', '▁i', '▁knew', '▁what', '▁the', '▁fu', 'ck', 'ing', 'hell', '▁is', '▁up', '▁with', '▁the', '▁thing', 'y']



```python
print(sp8_lower.EncodeAsPieces("Shittingduckcrappoopercrackingjack".lower()))
```

    ['▁shi', 't', 'ting', 'd', 'uck', 'c', 'ra', 'pp', 'oo', 'per', 'crack', 'ing', 'jack']



```python
print(sp8_lower.EncodeAsPieces("reaaaaaaaaaaaaaaaaaally".lower()))
```

    ['▁re', 'a', 'aaaa', 'aaaa', 'aaaa', 'aaaa', 'ally']



```python
print(sp8_lower.EncodeAsPieces('ElectricDildoInYourButt'.lower()))
```

    ['▁electric', 'd', 'il', 'do', 'in', 'y', 'our', 'but', 't']



```python
print(sp8_lower.EncodeAsPieces('There is value to simplicity\nWhich offers more explicitly\nSoundings more exquisitely\nWhen words are not too long\n'.lower()))
```

    ['▁there', '▁is', '▁value', '▁to', '▁simplicity', '\n', 'which', '▁offers', '▁more', '▁explicit', 'ly', '\n', 'sound', 'ing', 's', '▁more', '▁exquisite', 'ly', '\n', 'when', '▁words', '▁are', '▁not', '▁too', '▁long', '\n']



```python
df_trn = pd.read_csv(LM_PATH/'train_lower.csv', header=None, chunksize=chunksize)
df_val = pd.read_csv(LM_PATH/'test_lower.csv', header=None, chunksize=chunksize)
```


```python
#get_all_spm(sp32_lower, df_trn, 1)
```


```python
tok_trn_spm, trn_labels_spm = get_all_spm(sp8_lower, df_trn, 1)
tok_val_spm, val_labels_spm = get_all_spm(sp8_lower, df_val, 1)
```

    0
    1.89sec per 1k rows
    1
    2.34sec per 1k rows
    2
    2.30sec per 1k rows
    3
    2.14sec per 1k rows
    0
    1.95sec per 1k rows



```python
len(tok_trn_spm), len(tok_val_spm)
```




    (90000, 10000)




```python
(CLAS_PATH/'tmp').mkdir(exist_ok=True)

np.save(LM_PATH/'tmp'/'tok_trn_spm8_lower_ids.npy', tok_trn_spm)
np.save(LM_PATH/'tmp'/'tok_val_spm8_lower_ids.npy', tok_val_spm)

```

# now tokenize for classification


```python
df_trn = pd.read_csv(CLAS_PATH/'train_lower.csv', header=None, chunksize=chunksize)
df_val = pd.read_csv(CLAS_PATH/'test_lower.csv', header=None, chunksize=chunksize)
```


```python
tok_trn_spm, trn_labels_spm = get_all_spm(sp8_lower, df_trn, 1)
tok_val_spm, val_labels_spm = get_all_spm(sp8_lower, df_val, 1)
```

    0
    1.68sec per 1k rows
    1
    1.63sec per 1k rows
    0
    1.72sec per 1k rows
    1
    1.80sec per 1k rows



```python
len(tok_trn_spm), len(tok_val_spm)
```




    (25000, 25000)




```python
np.save(CLAS_PATH/'tmp'/'tok_trn_spm8_lower_ids.npy', tok_trn_spm)
np.save(CLAS_PATH/'tmp'/'tok_val_spm8_lower_ids.npy', tok_val_spm)

```


```python

# labels are no good. ignore. 
np.save(CLAS_PATH/'tmp'/'trn_labels_lower_spm8.npy', trn_labels_spm)
np.save(CLAS_PATH/'tmp'/'val_labels_lower_spm8.npy', val_labels_spm)
```
