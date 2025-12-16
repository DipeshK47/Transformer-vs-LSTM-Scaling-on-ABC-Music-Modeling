#!/bin/bash
set -e

cd /scratch/dk5288/code/my_project

# activate your env if needed
source /scratch/$USER/envs/cv6643/bin/activate

for size in tiny small medium large xl; do
  echo "=============================="
  echo "Starting training for $size"
  echo "=============================="
  python training_Transformer.py --model_size "$size"
  echo "Finished $size"
done

echo "All model sizes finished."