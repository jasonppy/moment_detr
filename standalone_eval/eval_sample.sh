#!/usr/bin/env bash
# Usage: bash standalone_eval/eval_sample.sh
# submission_path=standalone_eval/sample_val_preds.jsonl
# gt_path=data/highlight_val_release.jsonl
# save_path=standalone_eval/sample_val_preds_metrics.json

# PYTHONPATH=$PYTHONPATH:. python standalone_eval/eval.py \
# --submission_path ${submission_path} \
# --gt_path ${gt_path} \
# --save_path ${save_path}
detectfolder=${1} # e.g. scenedetect25

submission_path=/saltpool0/data/pyp/vqhighlight/submission/clip/${detectfolder}/submission.jsonl
gt_path=/home/pyp/vqhighlight/data/highlight_val_release_exist.jsonl
save_path=/home/pyp/moment_detr/standalone_eval/${detectfolder}_val_preds_metrics.json

python eval.py \
--submission_path ${submission_path} \
--gt_path ${gt_path} \
--save_path ${save_path}
