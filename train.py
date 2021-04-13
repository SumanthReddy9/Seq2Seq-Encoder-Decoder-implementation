import os
import math
import time
import spacy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataset import Dataset
from model import Encoder, Decoder, Seq2Seq
import config
from torchtext.data import Field, BucketIterator
from utils import bleu

def train(seq2seq, iterator, optimizer, criterion):
    seq2seq.train()

    epoch_loss = 0
    for batch in iterator:
        optimizer.zero_grad()
        outputs = seq2seq(batch.src, batch.trg)

        # 1. as mentioned in the seq2seq section, we will
        # cut off the first element when performing the evaluation
        # 2. the loss function only works on 2d inputs
        # with 1d targets we need to flatten each of them
        outputs_flatten = outputs[1:].view(-1, outputs.shape[-1])
        trg_flatten = batch.trg[1:].view(-1)
        loss = criterion(outputs_flatten, trg_flatten)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(seq2seq, iterator, criterion):
    seq2seq.eval()

    epoch_loss = 0
    with torch.no_grad():
        for batch in iterator:
            # turn off teacher forcing
            outputs = seq2seq(batch.src, batch.trg, teacher_forcing_ratio=0) 

            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]
            outputs_flatten = outputs[1:].view(-1, outputs.shape[-1])
            trg_flatten = batch.trg[1:].view(-1)
            loss = criterion(outputs_flatten, trg_flatten)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == "__main__":
    dataset = Dataset()
    train_data, valid_data, test_data, INPUT_DIM, OUTPUT_DIM = dataset.get_data()

    encoder = Encoder(INPUT_DIM, config.ENC_EMB_DIM, config.HID_DIM, config.N_LAYERS, config.ENC_DROPOUT)
    decoder = Decoder(OUTPUT_DIM, config.DEC_EMB_DIM, config.HID_DIM, config.N_LAYERS, config.DEC_DROPOUT)
    seq2seq = Seq2Seq(encoder, decoder, config.device).to(config.device)
    print(seq2seq)
    optimizer = optim.Adam(seq2seq.parameters())

    
    PAD_IDX = config.target.vocab.stoi['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    N_EPOCHS = 20
    best_valid_loss = float('inf')

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), batch_size=config.BATCH_SIZE, device=config.device)

    # for epoch in range(N_EPOCHS):    
    #     start_time = time.time()
    #     train_loss = train(seq2seq, train_iterator, optimizer, criterion)
    #     valid_loss = evaluate(seq2seq, valid_iterator, criterion)
    #     end_time = time.time()

    #     epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    #     if valid_loss < best_valid_loss:
    #         best_valid_loss = valid_loss
    #         torch.save(seq2seq.state_dict(), 'tut1-model.pt')

    #     # it's easier to see a change in perplexity between epoch as it's an exponential
    #     # of the loss, hence the scale of the measure is much bigger
    #     print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    #     print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    #     print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    
    seq2seq.load_state_dict(torch.load('tut1-model.pt'))

    test_loss = evaluate(seq2seq, test_iterator, criterion)

    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
    print("Bleu Score is :")
    print(bleu(test_data, seq2seq, config.source, config.target, config.device))
    # example_idx = 0
    # example = train_data.examples[example_idx]
    # print('source sentence: ', ' '.join(example.src))
    # print('target sentence: ', ' '.join(example.trg))

    # src_tensor = config.source.process([example.src]).to(config.device)
    # trg_tensor = config.target.process([example.trg]).to(config.device)

    # seq2seq.eval()
    # with torch.no_grad():
    #     outputs = seq2seq(src_tensor, trg_tensor, teacher_forcing_ratio=0)

    # output_idx = outputs[1:].squeeze(1).argmax(1)
    # result = ' '.join([config.target.vocab.itos[idx] for idx in output_idx])

    # print(result)