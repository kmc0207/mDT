import torch
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from keras.preprocessing.sequence import pad_sequences
import argparse
logger = logging.getLogger(__name__)



def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',dest='n_epoch',action='store_const',default=10)
    parser.add_argument('--data',dest='dataname',type=int,action='store_const',default='ag')
    parser.add_argument('--path',dest='path',type=str,action='store_const',required=True)
    parser.add_argument('--dropout',dest='dropout',type=float,action='store_const',default=0.1)
    parser.add_argument('--is_student',dest='is_student',type=bool,action='store_const',default=False)
    parser.add_argument('--kd_type',dest='kd_type',type=str,action='store_const',default='No_kd')
    parser.add_argument('--max_len',dest='maxlen',type=int,action='store_const',default=128)



def train():
    MAXLEN=maxlen
    train_data = pd.read_csv(path+dataname+'/train.csv',sep=',')
    test_data = pd.read_csv(path+dataname+'/test.csv',sep=',')
    data_labels = train_data.label
    train_y = train_data[data_labels[0]]
    train_x = train_data[data_labels[2]]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_x_tokenized = []
    for i in train_x:
        x=tokenizer(i,return_tensor='pt',return_attention_mask=False)
        train_x_tokenized.append(pad_sequences(
            x['input_ids'],maxlen=MAXLEN,dtype='long',value=0,truncating='post',padding='post')
        )
    train_x_tokenized = torch.tensor(train_x_tokenized)
    train_x_tokenized = train_x_tokenized.view(-1,128)
    attention_mask = []
    for i in train_x_tokenized:
        att_mask = [int(token_id>0) for token_id in i]
        attention_mask.append(att_mask)
    attention_mask = torch.tensor(attention_mask)
    train_y = torch.tensor(train_y)
    val_dataset = torch.utils.data.TensorDataset(
        train_x_tokenized[:1000],train_y[:1000],attention_mask[:1000]
    )
    train_dataset = torch.utils.data.TensorDataset(
        train_x_tokenized[1000:],train_y[1000:],attention_mask[1000:]
    )

    if is_student == True:
        print('student!')
    else:
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_label =4).to('cuda')



