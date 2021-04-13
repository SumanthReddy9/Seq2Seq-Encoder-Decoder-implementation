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
import config
from dataset import Dataset

class Encoder(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int, hid_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

    def forward(self, src_batch: torch.LongTensor):
        embedded = self.embedding(src_batch) 
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

class Decoder(nn.Module):

    def __init__(self, output_dim: int, emb_dim: int, hid_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.out = nn.Linear(hid_dim, output_dim)

    def forward(self, trg: torch.LongTensor, hidden: torch.FloatTensor, cell: torch.FloatTensor):
        embedded = self.embedding(trg.unsqueeze(0))
        outputs, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.out(outputs.squeeze(0))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            'Hidden dimensions of encoder and decoder must be equal!'
        assert encoder.n_layers == decoder.n_layers, \
            'Encoder and decoder must have equal number of layers!'

    def forward(self, src_batch: torch.LongTensor, trg_batch: torch.LongTensor,
                teacher_forcing_ratio: float=0.5):

        max_len, batch_size = trg_batch.shape
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder's output
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden & cell state of the encoder is used as the decoder's initial hidden state
        hidden, cell = self.encoder(src_batch)

        trg = trg_batch[0]
        for i in range(1, max_len):
            prediction, hidden, cell = self.decoder(trg, hidden, cell)
            outputs[i] = prediction

            if random.random() < teacher_forcing_ratio:
                trg = trg_batch[i]
            else:
                trg = prediction.argmax(1)

        return outputs

if __name__ == "__main__":
    # encoder = Encoder(config.INPUT_DIM, config.ENC_EMB_DIM, config.HID_DIM, config.N_LAYERS, config.ENC_DROPOUT).to(config.device)
    # hidden, cell = encoder(test_batch.src)
    # hidden.shape, cell.shape

    # decoder = Decoder(config.OUTPUT_DIM, config.DEC_EMB_DIM, config.HID_DIM, config.N_LAYERS, config.DEC_DROPOUT).to(config.device)
    # prediction, hidden, cell = decoder(test_batch.trg[0], hidden, cell)
    # prediction.shape, hidden.shape, cell.shape
    dataset = Dataset()
    train_data, valid_data, test_data, INPUT_DIM, OUTPUT_DIM = dataset.get_data()

    encoder = Encoder(INPUT_DIM, config.ENC_EMB_DIM, config.HID_DIM, config.N_LAYERS, config.ENC_DROPOUT)
    decoder = Decoder(OUTPUT_DIM, config.DEC_EMB_DIM, config.HID_DIM, config.N_LAYERS, config.DEC_DROPOUT)
    seq2seq = Seq2Seq(encoder, decoder, config.device).to(config.device)
    print(seq2seq)