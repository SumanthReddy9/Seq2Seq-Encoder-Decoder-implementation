import torch
import spacy
from torchtext.data.metrics import bleu_score
import sys


def translate_sentence(model, sentence, french, english, device, max_length=50):
    spacy_fr = spacy.load("fr")

    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_fr(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens.insert(0, french.init_token)
    tokens.append(french.eos_token)

    text_to_indices = [french.vocab.stoi[token] for token in tokens]

    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [english.vocab.stoi["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]
    return translated_sentence[1:]


def bleu(data, model, french, english, device):
    targets = []
    outputs = []
    print("Hello")
    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, french, english, device)
        prediction = prediction[:-1]

        targets.append([trg])
        outputs.append(prediction)
    
    print("Hello2")
    return bleu_score(outputs, targets)
