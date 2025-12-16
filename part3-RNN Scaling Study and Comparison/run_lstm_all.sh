#!/bin/bash
set -e

cd /scratch/dk5288/code/my_project/part3

for size in tiny small medium large; do
  echo "=============================="
  echo "Running LSTM training for $size"
  echo "=============================="
  python train_rnn_scaling.py --model_size "$size"
done

echo "All LSTM models finished."