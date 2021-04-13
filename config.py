import spacy
import torch
from torchtext.data import Field, BucketIterator

spacy_fr = spacy.load('fr_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')


class Tokenize():
    
    @staticmethod
    def tokenize_fr(text):
        return [tok.text for tok in spacy_fr.tokenizer(text)][::-1]

    @staticmethod
    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128
source = Field(tokenize=Tokenize.tokenize_fr, init_token='<sos>', eos_token='<eos>', lower=True)
target = Field(tokenize=Tokenize.tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5