import torch
from datasets import load_dataset
from tokenizers import tokenizers
from tokenizers.models import WordLevel
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset


from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import random
random.seed(42)

from pathlib import Path
from mymodel import build_transformer

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "datasource": 'opus_books',
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

config = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

def get_or_build_tokenizer(config, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


# Build tokenizers
tokenizer_src = get_or_build_tokenizer(config,  config['lang_src'])
tokenizer_tgt = get_or_build_tokenizer(config,  config['lang_tgt'])

# load dataset:
# Load the tensors from the file

# batch size = 2
# seq_len = 25
# embed_dim = d_model = 10
# src vocab size = 10248
# tgt vocab size = 24345

print(f"Load dataset info:--------------------------->")
loaded_tensors = torch.load('input_model_tensors2.pth')

# Access each tensor
encoder_input = loaded_tensors['encoder_input']
decoder_input = loaded_tensors['decoder_input']
encoder_mask = loaded_tensors['encoder_mask']
decoder_mask = loaded_tensors['decoder_mask']
label = loaded_tensors['label']

print("Loaded Tensors:")

print(f"{encoder_input.shape=}")
print(f"{encoder_mask.shape=}")
print(f"{decoder_input.shape=}")
print(f"{decoder_mask.shape=}")
print(f"{label.shape}")



en_vocab_size = tokenizer_src.get_vocab_size()
it_vocab_size = tokenizer_tgt.get_vocab_size()

model = build_transformer(src_vocab_size= en_vocab_size,
                          src_seq_len= 25,
                          tgt_vocab_size= it_vocab_size,
                          tgt_seq_len= 25,
                          d_model= 10,
                          d_ff = 60,
                          N=1,
                          dropout=0.1,
                          h = 2
                          )

encoder_output = model.encode(encoder_input, encoder_mask)
decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
proj_output = model.project(decoder_output)

print(f"Forward Pass:------------------------->")
print(f"{encoder_output.shape=}")
print(f"{decoder_output.shape=}")
print(f"{proj_output.shape=}")


# loss calculation :
#--------------------------------
print(f"Loss Calculation:----------------->")
print(f"decoder output: {proj_output.shape}") # batch, seq_len, vocab_size
print(f"true label shape: {label.shape}")   # bach seq_len(true values)

print(f"changed proj out shape: {proj_output.view(-1, tokenizer_tgt.get_vocab_size()).shape}")
print(f"changed label shape: {label.reshape(-1).shape}")

loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.reshape(-1))

print(f"Loss: {loss}")  


#########################################

