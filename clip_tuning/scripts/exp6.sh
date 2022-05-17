#!/bin/sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf2
export CUDA_VISIBLE_DEVICES=4,5


python \
../train.py \
--freeze_first_x 6 \
--seed 1 \
--opt_level O1 \
--num_workers 8 \
--exp_dir /saltpool0/scratch/pyp/vqhighlight/clip_tuning/exp6 \
--train_framerate 1 \
--val_framerate 1 \
--n_epochs 20 \
--n_print_steps 50 \
--n_val_steps 100 \
--logit_scale 0.07 \
--batch_size 96 \
--val_batch_size 1 \
--lr 0.000001 \
--warmup_fraction 0.1 \
--num_positives 6 \
--in_video_negatives 30 \
--in_batch_negatives 200 \
--alpha 53 >> ./logs/exp6.log 2>&1
# --alpha 53


