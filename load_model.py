import argparse
import sys
import torch
from torch import cuda
import torchtext.data as data
from classifier.train import predict, save
from classifier.mydatasets import DataSet
from src.vocabulary import Vocabulary
from src.style_transfer import StyleTransfer
from scripts.train_model import loadParams

#import classifier.main

parser = argparse.ArgumentParser()
parser.add_argument("--train_file_style1", type=str)
parser.add_argument("--train_file_style2", type=str)
parser.add_argument("--evaluation_file_style1", type=str)
parser.add_argument("--evaluation_file_style2", type=str)
parser.add_argument("--vocabulary", type=str)
parser.add_argument("--savefile", type=str)
parser.add_argument("--logdir", type=str, default="")
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
args = parser.parse_args()
print(args)
params = loadParams()
params.savefile = args.savefile
params.logdir = args.logdir

# Load Vocab
vocab = Vocabulary()
vocab.loadVocabulary(args.vocabulary)
vocab.initializeEmbeddings(params.embedding_size)

model = StyleTransfer(params, vocab)
model.load_state_dict(torch.load('model.pt'))
print('\nLoading model from {}...'.format(args.savefile))

# Data
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)

train_data, dev_data = DataSet.splits(
        text_field, label_field, root='data/all/')
text_field.build_vocab(train_data, dev_data)
label_field.build_vocab(train_data, dev_data)


# model.eval()
# print("Model Eval Ran Successfuly")


# def predict(text, model, text_field, label_feild, cuda_flag):
#     assert isinstance(text, str)
#     model.eval()
#     # text = text_field.tokenize(text)
#     text = text_field.preprocess(text)
#     text = [[text_field.vocab.stoi[x] for x in text]]
#     print(text)
#     print(type(text_field))
#     #print(text.tensor_type)
#     #x = text_field.tensor_type(text)
#     x = type(text_field)
#     torch.set_grad_enabled(False)   
#     if cuda_flag:
#         x = x.cuda()
#     print(x)
#     output = model(x)
#     _, predicted = torch.max(output, 1)
#     #return label_feild.vocab.itos[predicted.data[0][0]+1]
#     return label_feild.vocab.itos[predicted.data[0]+1]


sents = ["I am the president of this country.", "I think Obama is a great president", "The state of the union is strong."]
for sent in sents:
    label = predict(sent, model, text_field, label_field, cuda.is_available())
    print('\n[Text]  {}\n[Label] {}\n'.format(sent, label))


# model = torch.load('/Users/schen1337/Documents/text_style_transfer/data/models/yelp/experiment_0_models/model-2020-04-01-epoch_9-loss_16.367482.pt')
#model = torch.load('data/models/yelp/experiment_0_models/model-2020-04-01-epoch_9-loss_16.367482.pt')
#model = torch.load('model.pt')
