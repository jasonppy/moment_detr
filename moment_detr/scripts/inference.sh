#!/bin/sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf2
export CUDA_VISIBLE_DEVICES=0

ckpt_path=/saltpool0/scratch/pyp/vqhighlight/results/hl-video_tef-orig_no_saliency-2022_04_06_09_30_02/model_best.ckpt
eval_split_name=test # or test
eval_path=/home/pyp/vqhighlight/data/highlight_${eval_split_name}_release_exist.jsonl
PYTHONPATH=$PYTHONPATH:. python moment_detr/inference.py \
--resume ${ckpt_path} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path} \
${@:3} >> "/home/pyp/moment_detr/moment_detr/scripts/logs/inference_${eval_split}.log" 2>&1
