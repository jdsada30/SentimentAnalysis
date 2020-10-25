# -*- coding: utf-8 -*-
"""LSTM (1).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WeIjF6Nk-YeATWAOxnMqzAtjYnjbaxU-
"""

# Commented out IPython magic to ensure Python compatibility.

import torch
from torchtext import data

SEED = 1234
import pandas as pd
import numpy as np
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchtext


import random
from sklearn.metrics import classification_report

# %matplotlib inline

from torchnlp.nn import Attention


from utils.datautils import prepare_data
prepare_data('train_pos_full.txt', 'train_neg_full.txt', 'data')



import spacy
spacy_en = spacy.load('en_core_web_sm')

TEXT = data.Field(tokenize = 'spacy', include_lengths = True)
LABEL = data.LabelField(dtype = torch.float)

is_cuda = torch.cuda.is_available()
print("Cuda Status on system is {}".format(is_cuda))

# loading train, test and validation data 
train_data, valid_data, test_data = data.TabularDataset.splits(
    path="./", train="data/train.csv", 
    validation="data/valid.csv", test="data/test.csv",format="csv", skip_header=True, 
    fields=[('text', TEXT), ('label', LABEL)]
)



TEXT.build_vocab(train_data, vectors="glove.twitter.27B.100d", max_size=20000000, min_freq=5, unk_init = torch.Tensor.normal_ )
LABEL.build_vocab(train_data)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE,
    sort_within_batch = True,
    sort_key = lambda x: len(x.text),
    device = device)



class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        
        super().__init__()
       
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)

        self.att = Attention(hidden_dim)
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        # print(text.shape)
        #text = [sent len, batch size]
        
        embedded = self.dropout(self.embedding(text))
        
        # print(embedded.shape)
        #embedded = [sent len, batch size, emb dim]
        
        #pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
       
        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # print(hidden.shape)
        
        #unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        #output = [sent len, batch size, hid dim * num directions]
        #output over padding tokens are zero tensors
        
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]
        
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout

        hidden = hidden.transpose(0, 1).contiguous()
        
        # [batch_size, num_layers * num directions, hidden dim]

        hidden, weights = self.att(hidden, hidden)


        hidden = self.dropout(torch.cat((hidden[:,-2,:], hidden[:,-1,:]), dim = 1))
                
        #hidden = [batch size, hid dim * num directions]


            
        return self.fc(hidden)



INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = RNN(INPUT_DIM, 
            EMBEDDING_DIM, 
            HIDDEN_DIM, 
            OUTPUT_DIM, 
            N_LAYERS, 
            BIDIRECTIONAL, 
            DROPOUT, 
            PAD_IDX)





pretrained_embeddings = TEXT.vocab.vectors



model.embedding.weight.data.copy_(pretrained_embeddings)



UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)



import torch.optim as optim

optimizer = optim.Adam(model.parameters())

criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        text, text_lengths = batch.text
        
        
        predictions = model(text, text_lengths).squeeze(1)
        
        loss = criterion(predictions, batch.label)
        
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            text, text_lengths = batch.text
    
            predictions = model(text, text_lengths).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 15

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    f = open("logout_attention.txt", "a")
    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'model-lstmattention.pt')
    
    
    
    f.write(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    f.write(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    f.write(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%\n')
    f.close()


def save_vocab(vocab, path):
    with open(path, 'w+') as f:     
        for token, index in vocab.stoi.items():
            f.write(f'{index}\t{token}')

save_vocab(TEXT.vocab,"./vocab.txt")

vocab_dict = {}
for token, index in TEXT.vocab.stoi.items():
            vocab_dict[token] =  index

import pickle
f = open("file.pkl","wb")
pickle.dump(vocab_dict,f)
f.close()

len(TEXT.vocab)





"""Now predict competition data"""

import spacy
nlp = spacy.load('en_core_web_sm')

def predict_sentiment(model, sentence):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    prediction = torch.sigmoid(model(tensor, length_tensor))
    return prediction.item()



test_sentences = []
with open('data/test_data.txt') as file:
    for line in file:
        tweet= line.split(",")[0]
        tweet = line[len(tweet)+1:]
        test_sentences.append(tweet)

competition_data = pd.DataFrame({"tweet":test_sentences})
competition_data

predictions = []
for tweet in competition_data.tweet:
  predictions.append(predict_sentiment(model,tweet))

np.round(predictions)

output_arr=[]
id_count= 1
for pred in predictions:
    if(np.round(pred ) == 0):
        output_arr.append([id_count, -1])
    elif (np.round(pred ) == 1):
        output_arr.append([id_count, 1])
    id_count+=1

output_arr

output_df = pd.DataFrame(np.array(output_arr))
output_df.columns=["Id", "Prediction"]
output_df.set_index('Id', inplace=True)
output_df.to_csv("Predictions-lstmattention.csv")
