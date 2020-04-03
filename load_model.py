import argparse
import torch
from torch import cuda
from classifier.train import predict, save
from src.vocabulary import Vocabulary
from src.style_transfer import StyleTransfer
from scripts.train_model import loadParams

model = torch.load('/Users/schen1337/Documents/text_style_transfer/data/models/yelp/experiment_0_models/model-2020-04-01-epoch_9-loss_16.367482.pt')
model.eval()

sents = []
for sent in sents:
    predict(sent, model, text_field, label_feild, cuda.is_available())