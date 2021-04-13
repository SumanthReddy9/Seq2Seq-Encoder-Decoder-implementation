import os
import math
import time
import spacy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from typing import List
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from torchnlp.download import download_files_maybe_extract
import config
import itertools
from config import Tokenize
# from random import random

SEED = 2222
random.seed(SEED)
torch.manual_seed(SEED)


class Dataset():

    def get_data(self):
        source = config.source
        target = config.target
        train_data, valid_data, test_data = Multi30k.splits(exts=('.fr', '.en'), fields=(source, target), root="data/")
        l = len(train_data)
        train_data, test_data = train_data.split(split_ratio = 0.5, random_state=random.seed(SEED))
        train_data, valid_data = train_data.split(split_ratio=0.8, random_state=random.seed(SEED))

        # train_data, valid_data, test_data = self.multi30k_dataset(train=True, dev=True, test=True)
        # total_data = train_data + valid_data + test_data
        # train_data, valid_data, test_data = [], [], []
        # for data in total_data:
        #     _random = random.random()
        #     if _random<0.4:
        #         train_data.append(data)
        #     elif _random<0.5:
        #         valid_data.append(data)
        #     else:
        #         test_data.append(data)
        # for x in train_data[0:2]:
        #     print(x.src)

        # for data in total_data:
        #     data["en"] = Tokenize.tokenize_en(data['en'])
        #     data["de"] = Tokenize.tokenize_de(data['de'])
        # train_data, valid_data, test_data = total_data[0:int(len(total_data)*0.4)], total_data[int(len(total_data)*0.4):int(len(total_data)*0.5)], total_data[int(len(total_data)*0.5):]
        print("train", len(train_data), round(100*len(train_data)/l), "%")
        print("val", len(valid_data), round(100*len(valid_data)/l), "%")
        print("test", len(test_data), round(100*len(test_data)/l), "%")
        
        

        source.build_vocab(train_data, min_freq=2)
        target.build_vocab(train_data, min_freq=2)
        print(f"Unique tokens in source (de) vocabulary: {len(source.vocab)}")
        print(f"Unique tokens in target (en) vocabulary: {len(target.vocab)}")

        return train_data, valid_data, test_data, len(source.vocab), len(target.vocab)

if __name__ == "__main__":
    print('Helo')
    dataset  = Dataset()
    dataset.get_data()
