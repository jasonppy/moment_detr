#!/bin/sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf2
export CUDA_VISIBLE_DEVICES=0,1,2,3


python \
../train.py \
--seed 1 \
--opt_level O1 \
--num_workers 8 \
--exp_dir /saltpool0/scratch/pyp/vqhighlight/clip_tuning/exp1 \
--train_framerate 1 \
--val_framerate 1 \
--n_epochs 20 \
--n_print_steps 1 \
--n_val_steps 2 \
--logit_scale 0.07 \
--batch_size 32 \
--val_batch_size 1 \
--lr 0.00001 \
--warmup_fraction 0.1 \
--num_positives 5 \
--in_video_negatives 50 \
--in_batch_negatives 50 \
--alpha 53 >> ./logs/exp1.log 2>&1


