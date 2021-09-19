#!/bin/bash

seeds=(123 321 111 222 333)

datasets=("restaurant" "laptop")
sizes=("SMALL" "MEDIUM" "LARGE" "FULL")

for seed in ${seeds[@]};
do
  for dataset in ${datasets[@]}
  do
    for size in ${sizes[@]}
    do
        echo $seed
        echo $dataset
        echo $size
        python train_baseline.py --dataset $dataset --lm nolabel_prototype --seed $seed --size $size --name NOLABEL-GLOVE-SEED$seed-$size
    done
  done
done
