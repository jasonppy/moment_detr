#!/bin/sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf2
export CUDA_VISIBLE_DEVICES=0

dset_name=hl
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
results_root=/saltpool0/scratch/pyp/vqhighlight/results
exp_id=orig_with_saliency

######## data paths
train_path=/home/pyp/vqhighlight/data/highlight_train_release.jsonl
eval_path=/home/pyp/vqhighlight/data/highlight_val_release.jsonl
eval_split_name=val

######## setup video+text features
feat_root=/saltpool0/data/pyp/vqhighlight/features

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/slowfast_features)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_features)
  (( v_feat_dim += 512 ))
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/clip_text_features/
  t_feat_dim=512
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

#### training
bsz=32


PYTHONPATH=$PYTHONPATH:. python moment_detr/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
${@:1} >> "/home/pyp/moment_detr/moment_detr/scripts/logs/${exp_id}.log" 2>&1
