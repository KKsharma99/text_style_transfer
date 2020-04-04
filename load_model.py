import argparse
import torch
from torch import cuda
from classifier.train import predict, save
from src.vocabulary import Vocabulary
from src.style_transfer import StyleTransfer
from scripts.train_model import loadParams

parser = argparse.ArgumentParser()
parser.add_argument("--train_file_style1", type=str)
parser.add_argument("--train_file_style2", type=str)
parser.add_argument("--evaluation_file_style1", type=str)
parser.add_argument("--evaluation_file_style2", type=str)
parser.add_argument("--vocabulary", type=str)
parser.add_argument("--savefile", type=str)
parser.add_argument("--logdir", type=str, default="")
args = parser.parse_args()

params = loadParams()
params.savefile = args.savefile
params.logdir = args.logdir
vocab = Vocabulary()
vocab.loadVocabulary(args.vocabulary)
vocab.initializeEmbeddings(params.embedding_size)

model = StyleTransfer(params, vocab)
model.load_state_dict(torch.load('model.pt'))
# model = torch.load('/Users/schen1337/Documents/text_style_transfer/data/models/yelp/experiment_0_models/model-2020-04-01-epoch_9-loss_16.367482.pt')
#model = torch.load('data/models/yelp/experiment_0_models/model-2020-04-01-epoch_9-loss_16.367482.pt')
#model = torch.load('model.pt')
model.eval()
print("Model Eval worked")
sents = []
for sent in sents:
    predict(sent, model, text_field, label_feild, cuda.is_available())
