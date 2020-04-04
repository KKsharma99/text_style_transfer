#!/bin/bash

python -m scripts.train_model \
  --train_file_style1 data/yelp/train/positive.txt \
  --train_file_style2 data/yelp/train/negative.txt \
  --evaluation_file_style1 data/yelp/dev/positive.txt \
  --evaluation_file_style2 data/yelp/dev/negative.txt \
  --vocabulary data/yelp/vocabulary.pickle \
  --savefile data/models/yelp/model \
  --logdir data/models/yelp/log/