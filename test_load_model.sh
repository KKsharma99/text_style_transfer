#!/bin/bash

python3 -m load_model \
  --beam_width 16 \
  --temperature 8 \
  --max_len 2 \
  --train_file_style1 data/yelp/train/positive.txt \
  --train_file_style2 data/yelp/train/negative.txt \
  --evaluation_file_style1 data/yelp/dev/positive.txt \
  --evaluation_file_style2 data/yelp/dev/negative.txt \
  --vocabulary data/yelp/vocabulary.pickle \
  --savefile data/models/yelp/model \
  --logdir data/models/yelp/log/
