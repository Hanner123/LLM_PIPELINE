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
from bert_dataset import BERTDataset, PositionalEmbedding, BERTEmbedding, MultiHeadedAttention, FeedForward, EncoderLayer, BERT, NextSentencePrediction, MaskedLanguageModel, BERTLM, ScheduledOptim, BERTTrainer
import shutil

MAX_LEN = 64
#https://medium.com/data-and-beyond/complete-guide-to-building-bert-model-from-sratch-3e6562228891 tutorial
### loading all data into memory
corpus_movie_conv = '../datasets/movie_conversations.txt'
corpus_movie_lines = '../datasets/movie_lines.txt'
with open(corpus_movie_conv, 'r', encoding='iso-8859-1') as c:
    conv = c.readlines()
with open(corpus_movie_lines, 'r', encoding='iso-8859-1') as l:
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
pairs = pairs[:10000] #für testlauf


# WordPiece tokenizer

### save data as txt file
text_data = []
file_count = 0

for sample in tqdm.tqdm([x[0] for x in pairs]):
    text_data.append(sample)

    # once we hit the 10K mark, save to file
    if len(text_data) == 10000:
        with open(f'../data/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(text_data))
        text_data = []
        file_count += 1

paths = [str(x) for x in Path('../data').glob('**/*.txt')]

### training own tokenizer
tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=True
)

tokenizer.train( 
    files=paths,
    vocab_size=30_000, 
    min_frequency=5,
    limit_alphabet=1000, 
    wordpieces_prefix='##',
    special_tokens=['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
    )
if os.path.exists('../bert-it-1'):
    shutil.rmtree('../bert-it-1')
os.mkdir('../bert-it-1')
tokenizer.save_model('../bert-it-1', 'bert-it')
tokenizer = BertTokenizer.from_pretrained('../bert-it-1/bert-it-vocab.txt', local_files_only=True)

train_data = BERTDataset(
   pairs, seq_len=MAX_LEN, tokenizer=tokenizer)
train_loader = DataLoader(
   train_data, batch_size=32, shuffle=True, pin_memory=True)
sample_data = next(iter(train_loader))
print(train_data[random.randrange(len(train_data))])

print("testlauf:....") #2h pro Epoche -> ich muss was kürzen damit es nur 1/2 h dauert
train_data = BERTDataset(
   pairs, seq_len=MAX_LEN, tokenizer=tokenizer)

train_loader = DataLoader(
   train_data, batch_size=32, shuffle=True, pin_memory=True)

bert_model = BERT(
  vocab_size=len(tokenizer.vocab),
  d_model=768,
  n_layers=2,
  heads=12,
  dropout=0.1
)

bert_lm = BERTLM(bert_model, len(tokenizer.vocab))
bert_trainer = BERTTrainer(bert_lm, train_loader, device='cpu')
epochs = 5

for epoch in range(epochs):
  bert_trainer.train(epoch)

  