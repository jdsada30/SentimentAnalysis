import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class CharDataset(Dataset):
    def __init__(self, filename, vocab, max_length=140, data_dir=''):
        self.data_path = os.path.join(data_dir, filename)
        self.vocabulary = list(vocab); 
        self.identity_mat = np.identity(len(self.vocabulary))
        
        tweets = [] 
        df = pd.read_csv(self.data_path)
        for tweet in df.Tweet.values:
          tweet_char = ""
          for char in tweet:
            tweet_char += char
            tweet_char += " "
          tweets.append(tweet_char)
          
        self.tweets = tweets
        self.labels = df.Label.values
        self.max_length = max_length
        self.length = len(self.labels)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        tweet = self.tweets[index]
        char_embedding = np.array([self.identity_mat[self.vocabulary.index(i)] for i in list(tweet) if i in self.vocabulary],
                        dtype=np.float32)
        if len(char_embedding) > self.max_length:
            char_embedding = char_embedding[:self.max_length]
        elif 0 < len(char_embedding) < self.max_length:
            char_embedding = np.concatenate(
                (char_embedding, np.zeros((self.max_length - len(char_embedding), len(self.vocabulary)), dtype=np.float32)))
        elif len(char_embedding) == 0:
            char_embedding = np.zeros((self.max_length, len(self.vocabulary)), dtype=np.float32)
        label = self.labels[index]
        return char_embedding, label
