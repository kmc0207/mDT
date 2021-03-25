import torch
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import logging
import argparse



def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',dest='n_epoch',default=10)
    parser.add_argument('--data',dest='dataname',type=str,default='ag')
    parser.add_argument('--path',dest='path',type=str,required=True)
    parser.add_argument('--dropout',dest='dropout',type=float,default=0.1)
    parser.add_argument('--is_student',dest='is_student',type=bool,default=False)
    parser.add_argument('--kd_type',dest='kd_type',type=str,default='No_kd')
    parser.add_argument('--max_len',dest='maxlen',type=int,default=128)
    args = parser.parse_args()
    return args

def validation(model,val_dataset):
  model.eval()
  acc=0
  total_len = 0
  for x,y,mask in val_dataset:
    x=x.to('cuda')
    y=y-1
    leng=x.size()[0]
    y=y.to('cuda')
    mask =mask.to('cuda')
    with torch.no_grad():
      outputs = model(x,attention_mask=mask)
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    y = y.to('cpu').numpy()
    tmp_acc,tmp_len = flat_accuracy(logits,y)
    acc += tmp_acc
    total_len += tmp_len
  return acc*100/total_len

def flat_accuracy(preds,labels):
  preds_flat = np.argmax(preds,axis=1).flatten()
  labels_flat = labels.flatten()

  return np.sum(preds_flat == labels_flat),len(labels_flat)


def train(args):
    MAXLEN=args.maxlen
    train_data = pd.read_csv(args.path+'/'+args.dataname+'/train.csv',sep=',')
    test_data = pd.read_csv(args.path+'/'+args.dataname+'/test.csv',sep=',')
    data_labels = train_data.columns
    train_y = train_data[data_labels[0]]
    train_x = train_data[data_labels[2]]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_x_tokenized = []
    attention_mask = []
    for i in train_x:
        x=tokenizer(
                i,
                return_tensors='pt',
                return_attention_mask=True,
                padding='max_length',
                truncation=True,
                max_length=MAXLEN
                )
        train_x_tokenized.append(x['input_ids'])
        attention_mask.append(x['attention_mask'])
    train_x_tokenized = torch.stack(train_x_tokenized)
    train_x_tokenized = train_x_tokenized.view(-1,MAXLEN)
    mylogger.info('finish tokenize')
    attention_mask = torch.stack(attention_mask)
    train_y = torch.tensor(train_y)
    val_dataset = torch.utils.data.TensorDataset(
        train_x_tokenized[:1000],train_y[:1000],attention_mask[:1000]
    )
    train_dataset = torch.utils.data.TensorDataset(
        train_x_tokenized[1000:],train_y[1000:],attention_mask[1000:]
    )
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64,shuffle=False)
    dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=64,shuffle=False)
    if args.is_student == True:
        print('student!')
    else:
        mylogger.info('train a teacher')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels =4).to('cuda')
        optim = torch.optim.Adam(model.parameters(),lr=1e-5,eps=1e-8)
        mylogger.info('start training')
        for epoch in range(args.n_epoch):
            losses = 0
            for x,y,mask in dataloader:
                optim.zero_grad()
                y=y-1
                mask = mask.to('cuda')
                x=x.to('cuda')
                y=y.to('cuda')
                output=model(x,labels=y,attention_mask=mask)
                loss=outpu[0]
                losses += loss.item()
                loss.backward()
                optim.step()
            mylogger.info('epoch : {}/{} done, train_loss{:.2f},val_acc{:.2f}'.format(
                epoch+1,args.n_epoch,losses,validation(model,val_dataloader)))


args = get_argument()
mylogger = logging.getLogger("my")
mylogger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s-%(message)s')
stream_hander = logging.StreamHandler()
stream_hander.setFormatter(formatter)
mylogger.addHandler(stream_hander)
file_handler = logging.FileHandler(args.path+'/train.log')
mylogger.addHandler(file_handler)
mylogger.info('start!')
train(args)


