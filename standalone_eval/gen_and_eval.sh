#!/usr/bin/env bash
# Usage: bash standalone_eval/eval_sample.sh
# submission_path=standalone_eval/sample_val_preds.jsonl
# gt_path=data/highlight_val_release.jsonl
# save_path=standalone_eval/sample_val_preds_metrics.json

# PYTHONPATH=$PYTHONPATH:. python standalone_eval/eval.py \
# --submission_path ${submission_path} \
# --gt_path ${gt_path} \
# --save_path ${save_path}
# detectfolder=${1} # e.g. scenedetect25

detectfolders=("scenedetect20" "scenedetect25" "scenedetect27" "scenedetect30" "scenedetect32" "scenedetect35" "scenedetect38" "scenedetect41" "scenedetect44" "scenedetect47")
detectfolders=("uniform")
echo "generate similarity scores"
# python gen_clip_submission.py --scenedetect_folder $detectfolder
# # python gen_clip_submission.py --scenedetect_folder $detectfolder --fixed_scenes
# python gen_clip_submission.py --scenedetect_folder $detectfolder --sliding_window --window_size 2 --stride 2 --watershed --gammas 0.01 0.05 0.1 .15 0.25 .4 .5 .6 .7 .8 .9 .95 --taus 0.05 .1 .2 .3 .4 .5 .6 0.8 1.0 --iou_threshold 0.99
# python gen_clip_submission.py --scenedetect_folder $detectfolder --watershed --watershed --gammas 0.01 0.05 0.1 .15 0.25 .4 .5 .6 .7 .8 .9 .95 --taus 0.05 .1 .2 .3 .4 .5 .6 0.8 1.0 --iou_threshold 0.99


echo "evaluation"
submission_path=/saltpool0/data/pyp/vqhighlight/submission/clip/${detectfolder}/submission.jsonl
gt_path=/home/pyp/vqhighlight/data/highlight_val_release_exist.jsonl
save_path=/home/pyp/moment_detr/standalone_eval/${detectfolder}_val_preds_metrics.json

python eval.py \
--submission_path ${submission_path} \
--gt_path ${gt_path} \
--save_path ${save_path} >> "./logs/clip_watershed.log" 2>&1


for detectfolder in ${detectfolders[@]}; do
    echo "generate similarity scores"
    python gen_clip_submission.py --scenedetect_folder $detectfolder
    # # python gen_clip_submission.py --scenedetect_folder $detectfolder --fixed_scenes
    # python gen_clip_submission.py --scenedetect_folder $detectfolder --sliding_window --window_size 2 --stride 2 --watershed --gammas 0.01 0.05 0. .15 0.25 .4 .5 .6 .7 .8 .9 .95 --taus 0.05 .1 .2 .3 .4 .5 .6 0.8 1.0 --iou_threshold 0.99
    # python gen_clip_submission.py --scenedetect_folder $detectfolder --watershed --watershed --gammas 0.01 0.05 0.1 .15 0.25 .4 .5 .6 .7 .8 .9 .95 --taus 0.05 .1 .2 .3 .4 .5 .6 0.8 1.0 --iou_threshold 0.99


    echo "evaluation"
    submission_path=/saltpool0/data/pyp/vqhighlight/submission/clip/${detectfolder}/submission.jsonl
    gt_path=/home/pyp/vqhighlight/data/highlight_val_release_exist.jsonl
    save_path=/home/pyp/moment_detr/standalone_eval/${detectfolder}_val_preds_metrics.json

    python eval.py \
    --submission_path ${submission_path} \
    --gt_path ${gt_path} \
    --save_path ${save_path} >> "./logs/fixed_original_${detectfolder}.log" 2>&1
    # --save_path ${save_path}
done

detectfolders=("scenedetect35" "scenedetect38" "scenedetect41" "scenedetect44" "scenedetect47")
# detectfolders=("scenedetect35" "scenedetect38" "scenedetect41" "scenedetect44" "scenedetect47")
# detectfolders=("scenedetect25")

for detectfolder in ${detectfolders[@]}; do
    echo "generate similarity scores"
    # # python gen_clip_submission.py --scenedetect_folder $detectfolder
    python gen_clip_submission.py --scenedetect_folder $detectfolder --fixed_scenes
    # python gen_clip_submission.py --scenedetect_folder $detectfolder --sliding_window --window_size 2 --stride 2 --watershed --gammas 0.01 0.05 0. .15 0.25 .4 .5 .6 .7 .8 .9 .95 --taus 0.05 .1 .2 .3 .4 .5 .6 0.8 1.0 --iou_threshold 0.99
    # python gen_clip_submission.py --scenedetect_folder $detectfolder --watershed --watershed --gammas 0.01 0.05 0.1 .15 0.25 .4 .5 .6 .7 .8 .9 .95 --taus 0.05 .1 .2 .3 .4 .5 .6 0.8 1.0 --iou_threshold 0.99


    echo "evaluation"
    submission_path=/saltpool0/data/pyp/vqhighlight/submission/clip/${detectfolder}/submission.jsonl
    gt_path=/home/pyp/vqhighlight/data/highlight_val_release_exist.jsonl
    save_path=/home/pyp/moment_detr/standalone_eval/${detectfolder}_val_preds_metrics.json

    python eval.py \
    --submission_path ${submission_path} \
    --gt_path ${gt_path} \
    --save_path ${save_path} >> "./logs/fixedScenes_${detectfolder}.log" 2>&1
    # --save_path ${save_path}
done


detectfolders=("scenedetect49" "scenedetect53" "scenedetect57")
detectfolders=("scenedetect63" "scenedetect69" "scenedetect72")
for detectfolder in ${detectfolders[@]}; do
    echo "generate similarity scores"
    python gen_clip_submission.py --scenedetect_folder $detectfolder
    # python gen_clip_submission.py --scenedetect_folder $detectfolder --fixed_scenes
    # python gen_clip_submission.py --scenedetect_folder $detectfolder --sliding_window --window_size 2 --stride 2 --watershed --gammas 0.01 0.05 0. .15 0.25 .4 .5 .6 .7 .8 .9 .95 --taus 0.05 .1 .2 .3 .4 .5 .6 0.8 1.0 --iou_threshold 0.99
    # python gen_clip_submission.py --scenedetect_folder $detectfolder --watershed --watershed --gammas 0.01 0.05 0.1 .15 0.25 .4 .5 .6 .7 .8 .9 .95 --taus 0.05 .1 .2 .3 .4 .5 .6 0.8 1.0 --iou_threshold 0.99


    echo "evaluation"
    submission_path=/saltpool0/data/pyp/vqhighlight/submission/clip/${detectfolder}/submission.jsonl
    gt_path=/home/pyp/vqhighlight/data/highlight_val_release_exist.jsonl
    save_path=/home/pyp/moment_detr/standalone_eval/${detectfolder}_val_preds_metrics.json

    python eval.py \
    --submission_path ${submission_path} \
    --gt_path ${gt_path} \
    --save_path ${save_path} >> "./logs/fixed_original_${detectfolder}.log" 2>&1
    # --save_path ${save_path}
done

detectfolders=("slidingwindow_scenes")
for detectfolder in ${detectfolders[@]}; do
    echo "generate similarity scores"
    python gen_clip_submission.py --scenedetect_folder $detectfolder
    # python gen_clip_submission.py --scenedetect_folder $detectfolder --fixed_scenes
    # python gen_clip_submission.py --scenedetect_folder $detectfolder --sliding_window --window_size 2 --stride 2 --watershed --gammas 0.01 0.05 0. .15 0.25 .4 .5 .6 .7 .8 .9 .95 --taus 0.05 .1 .2 .3 .4 .5 .6 0.8 1.0 --iou_threshold 0.99
    # python gen_clip_submission.py --scenedetect_folder $detectfolder --watershed --watershed --gammas 0.01 0.05 0.1 .15 0.25 .4 .5 .6 .7 .8 .9 .95 --taus 0.05 .1 .2 .3 .4 .5 .6 0.8 1.0 --iou_threshold 0.99


    echo "evaluation"
    submission_path=/saltpool0/data/pyp/vqhighlight/submission/clip/${detectfolder}/submission.jsonl
    gt_path=/home/pyp/vqhighlight/data/highlight_val_release_exist.jsonl
    save_path=/home/pyp/moment_detr/standalone_eval/${detectfolder}_val_preds_metrics.json

    python eval.py \
    --submission_path ${submission_path} \
    --gt_path ${gt_path} \
    --save_path ${save_path} >> "./logs/fixed_original_${detectfolder}.log" 2>&1
    # --save_path ${save_path}
done

detectfolders=("scenedetect35" "scenedetect38" "scenedetect41" "scenedetect44" "scenedetect47" "scenedetect49" "scenedetect53" "scenedetect57" "scenedetect63" "scenedetect69" "scenedetect72")
# detectfolders=("scenedetect35" "scenedetect38" "scenedetect41" "scenedetect44" "scenedetect47")
detectfolders=("scenedetect47")
# detectfolders=("scenedetect44" "scenedetect49" "scenedetect53" "scenedetect57" "scenedetect63" "scenedetect69")


for detectfolder in ${detectfolders[@]}; do
    echo "generate similarity scores"
    python upper_bound.py --scenedetect_folder $detectfolder
    

    echo "evaluation"
    submission_path=/saltpool0/data/pyp/vqhighlight/submission/clip/${detectfolder}/submission.jsonl
    gt_path=/home/pyp/vqhighlight/data/highlight_val_release_exist.jsonl
    save_path=/home/pyp/moment_detr/standalone_eval/${detectfolder}_val_preds_metrics.json

    python eval.py \
    --submission_path ${submission_path} \
    --gt_path ${gt_path} \
    --save_path ${save_path} >> "./logs/upper_bound_${detectfolder}.log" 2>&1
done


# detectfolders=("scenedetect20" "scenedetect35" "scenedetect38" "scenedetect41" "scenedetect44" "scenedetect47" "scenedetect49" "scenedetect53" "scenedetect57" "scenedetect63" "scenedetect69" "scenedetect72")
# # detectfolders=("scenedetect35" "scenedetect38" "scenedetect41" "scenedetect44" "scenedetect47")
# detectfolders=("scenedetect47")
detectfolders=("scenedetect20" "scenedetect25" "scenedetect27" "scenedetect30" "scenedetect32" "scenedetect35" "scenedetect38" "scenedetect41" "scenedetect44" "scenedetect47" "scenedetect49" "scenedetect53" "scenedetect57" "scenedetect63" "scenedetect69" "scenedetect72")


# detectfolders=("scenedetect57")
for detectfolder in ${detectfolders[@]}; do
    echo "generate similarity scores"
    python upper_bound.py --scenedetect_folder $detectfolder --combine_units
    

    echo "evaluation"
    submission_path=/saltpool0/data/pyp/vqhighlight/submission/clip/${detectfolder}/submission.jsonl
    gt_path=/home/pyp/vqhighlight/data/highlight_val_release_exist.jsonl
    save_path=/home/pyp/moment_detr/standalone_eval/${detectfolder}_val_preds_metrics.json

    python eval.py \
    --submission_path ${submission_path} \
    --gt_path ${gt_path} \
    --save_path ${save_path} >> "./logs/upper_bound_combine_units_${detectfolder}.log" 2>&1
done




detectfolders=("scenedetect41" "scenedetect44" "scenedetect47" "scenedetect49" "scenedetect53" "scenedetect57" "scenedetect63")
# drop_thresholds=(0.2 0.22 0.24)
drop_thresholds=(0.26 0.28 0.30)
for detectfolder in ${detectfolders[@]}; do
    for drop_threshold in ${drop_thresholds[@]}; do
        echo "generate similarity scores"
        python gen_clip_submission.py --scenedetect_folder $detectfolder --drop_threshold ${drop_threshold}
        

        echo "evaluation"
        submission_path=/saltpool0/data/pyp/vqhighlight/submission/clip/${detectfolder}/submission.jsonl
        gt_path=/home/pyp/vqhighlight/data/highlight_val_release_exist.jsonl
        save_path=/home/pyp/moment_detr/standalone_eval/${detectfolder}_val_preds_metrics.json

        python eval.py \
        --submission_path ${submission_path} \
        --gt_path ${gt_path} \
        --save_path ${save_path} >> "./logs/fixed_original_drop${drop_threshold}_${detectfolder}.log" 2>&1
        # --save_path ${save_path}
    done
done
