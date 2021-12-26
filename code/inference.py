from model_wrapper import ModelForMaskedLM
from dataset import CorpusDataset, CorpusDatasetV2
from transformers import BertTokenizer, RobertaTokenizerFast, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import time
import os
import json
import pickle
import numpy as np
import argparse
import utils
import lz4.frame

parser = argparse.ArgumentParser()
parser.add_argument("--use_data", type = utils.str2bool, help = "use_data", dest = "use_data", default = False)
args = parser.parse_args()

with open("/content/Nystromformer/nystrom-512/config.json", "r") as f:
    config = json.load(f)

model_config = config["model"]
pretraining_config = config["pretraining_setting"]
gpu_config = config["gpu_setting"]
checkpoint_dir = config["model_checkpoints"]
data_folder = config["data_folder"]

########################### Loading Datasets ###########################

#tokenizer = RobertaTokenizerFast.from_pretrained('/code/roberta-base', model_max_length = model_config["max_seq_len"])
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer.model_max_length = model_config["max_seq_len"]
tokenizer.init_kwargs['model_max_length'] = model_config["max_seq_len"]
model_config["vocab_size"] = len(tokenizer.get_vocab())

########################### Loading Model ###########################

model = ModelForMaskedLM(model_config)

########################### Forward pass ###########################

text = "hello world"
encoding = tokenizer(text, return_tensors = "pt")

outputs = model(**encoding)






      
