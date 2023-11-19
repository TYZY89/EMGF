#!/bin/bash

# * laptop
#  python ./train.py --model_name EHFB --dataset laptop --seed 1000 --bert_lr 2e-5 --batch_size 8 --num_epoch 15 --bert_dim 768 --max_length 100 --vocab_dir ./dataset/Laptops_corenlp --gamma 1 --theta 0.12\
#                     --max_num_spans 6 --dep_layers 3 --sem_layers 6 --fusion_condition ResEMFH 

# * restaurants
 python ./train.py --model_name EHFB --dataset restaurant --seed 1000 --bert_lr 2e-5 --batch_size 16 --num_epoch 20 --vocab_dir ./dataset/Restaurants_corenlp --bert_dim 768 --max_length 100 \
                    --max_num_spans 3 --dep_layers 3 --sem_layers 9 --fusion_condition ResEMFH --gamma 1 --theta 0.12

# * twitter
# python ./train.py --model_name EHFB --dataset twitter --seed 1000 --bert_lr 2e-5 --batch_size 16 --num_epoch 20 --vocab_dir ./dataset/Tweets_corenlp --bert_dim 768 --max_length 100 \
#                   --max_num_spans 3 --dep_layers 3 --sem_layers 1 --fusion_condition ResEMFH --gamma 1 --theta 0.07


