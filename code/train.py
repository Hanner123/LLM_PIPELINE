import os
from pathlib import Path
import torch
import re
import random
import transformers, datasets
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
import tqdm
from torch.utils.data import Dataset, DataLoader
import itertools
import math
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from bert_dataset_quant import BERTDataset, PositionalEmbedding, BERTEmbedding, MultiHeadedAttention, FeedForward, EncoderLayer, BERT, NextSentencePrediction, MaskedLanguageModel, BERTLM, ScheduledOptim, BERTTrainer
import shutil
import yaml
import json

MAX_LEN = 64
#https://medium.com/data-and-beyond/complete-guide-to-building-bert-model-from-sratch-3e6562228891 tutorial
### loading all data into memory
movie_convs = Path(__file__).resolve().parent.parent / "datasets" / "movie_conversations.txt"
movie_lines = Path(__file__).resolve().parent.parent / "datasets" / "movie_lines.txt"

def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

params = load_params()
epochs = params["train"]["epochs"]

with open(movie_convs, 'r', encoding='iso-8859-1') as c:
    conv = c.readlines()
with open(movie_lines, 'r', encoding='iso-8859-1') as l:
    lines = l.readlines()

### splitting text using special lines
lines_dic = {}
for line in lines:
    objects = line.split(" +++$+++ ")
    lines_dic[objects[0]] = objects[-1]

### generate question answer pairs
pairs = []
for con in conv:
    ids = eval(con.split(" +++$+++ ")[-1])
    for i in range(len(ids)):
        qa_pairs = []
        
        if i == len(ids) - 1:
            break

        first = lines_dic[ids[i]].strip()  
        second = lines_dic[ids[i+1]].strip() 

        qa_pairs.append(' '.join(first.split()[:MAX_LEN]))
        qa_pairs.append(' '.join(second.split()[:MAX_LEN]))
        pairs.append(qa_pairs)


print("Pairs:", len(pairs))
# pairs = pairs[:10000] #für testlauf


# WordPiece tokenizer

### save data as txt file
text_data = []
file_count = 0


paths = []


for sample in tqdm.tqdm([x[0] for x in pairs]): #was ist das?
    text_data.append(sample)
    print("Text data:", sample)

    # once we hit the 10K mark, save to file
    if len(text_data) == 10000:
        filename = f'../data/text_{file_count}.txt'
        text_paths = Path(__file__).resolve().parent.parent / "data" / filename
        paths.append(str(text_paths))
        with open(text_paths, 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(text_data))
        text_data = []
        file_count += 1
# Save remaining data
if text_data:
    filename = f'text_{file_count}.txt'
    text_paths = Path(__file__).resolve().parent.parent / "data" / filename
    with open(text_paths, 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(text_data))




### training own tokenizer
tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=True
)

tokenizer.train( 
    files=paths,
    vocab_size=30000, 
    min_frequency=4,
    limit_alphabet=1000, 
    wordpieces_prefix='##',
    special_tokens=['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
    )


tokenizer_path = Path(__file__).resolve().parent.parent / "bert-it-1" / "vocab.txt"
path = Path(__file__).resolve().parent.parent / "bert-it-1"
if path.exists():
    shutil.rmtree(path)
os.mkdir(path)

tokenizer.save_model(str(path))



tokenizer = BertTokenizer.from_pretrained(str(path), local_files_only=True)

print("Vocab length tokenizer", len(tokenizer.vocab))
tokenizer_config = {
    "tokenizer_class": "BertTokenizer",
    "vocab_size": len(tokenizer.vocab),
    "do_lower_case": True
}

with open(path / "tokenizer_config.json", "w") as f:
    json.dump(tokenizer_config, f)

# train_data = BERTDataset(
#    pairs, seq_len=MAX_LEN, tokenizer=tokenizer)
# train_loader = DataLoader(
#    train_data, batch_size=32, shuffle=True, pin_memory=True)



print("testlauf:....")
subset_size = len(pairs) // 10
pairs_subset = pairs[:subset_size]

train_data = BERTDataset(
   pairs_subset, seq_len=MAX_LEN, tokenizer=tokenizer)


# Tokenized dataset für evaluation & msnt speichern
# all_tokenized = [train_data[i] for i in range(len(train_data))]
# tokenized_data = Path(__file__).resolve().parent.parent / "datasets" / "tokenized_pairs_bert.pt"
# torch.save(all_tokenized, tokenized_data)


train_loader = DataLoader(
   train_data, batch_size=32, shuffle=True, pin_memory=True)

print(train_data[random.randrange(len(train_data))])

sample_data = next(iter(train_loader))

print("Vocab length tokenizer", len(tokenizer.vocab))

bert_model = BERT(
  vocab_size=len(tokenizer.vocab),
  d_model=768,
  n_layers=2,
  heads=12,
  dropout=0.1
)

bert_lm = BERTLM(bert_model, len(tokenizer.vocab))


bert_trainer = BERTTrainer(bert_lm, train_loader, device='cpu')


for epoch in range(epochs):
  bert_trainer.train(epoch)

model_dir = Path(__file__).resolve().parent.parent / "models" / "bert.pth"
torch.save(bert_lm.state_dict(), model_dir)




