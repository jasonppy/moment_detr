import os
import numpy as np
from numpy import linalg as LA
import jsonlines
import argparse
import csv
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from watershed import watershed
# the goal is to generate the following format
# {"qid": 2579, "query": "A girl and her mother cooked while talking with each other on facetime.", "vid": "NUsG9BgSes0_210.0_360.0", "pred_relevant_windows": [[0.0, 70.0, 0.9986], [78.0, 146.0, 0.4138], [0.0, 146.0, 0.0444], [108.0, 150.0, 0.0364], [0.0, 48.0, 0.0081], [56.0, 116.0, 0.0071], [38.0, 82.0, 0.0006], [62.0, 88.0, 0.0006], [132.0, 146.0, 0.0001], [52.0, 88.0, 0.0]], "pred_saliency_scores": [-0.2452392578125, -0.3779296875, -0.474609375, -0.474365234375, -0.52880859375, -0.267578125, -0.27197265625, -0.19580078125, -0.2034912109375, -0.058135986328125, -0.2442626953125, -0.2142333984375, -0.02569580078125, -0.09539794921875, 0.0751953125, 0.073486328125, -0.009185791015625, 0.028411865234375, 0.060882568359375, 0.1505126953125, 0.1416015625, -0.04473876953125, 0.02642822265625, -0.041900634765625, 0.0595703125, 0.127685546875, 0.10394287109375, 0.22216796875, 0.07666015625, 0.02508544921875, 0.1064453125, 0.1470947265625, 0.11492919921875, 0.1197509765625, 0.1614990234375, 0.12890625, 0.0863037109375, -0.0022640228271484375, 0.1641845703125, 0.16259765625, -0.02667236328125, -0.90771484375, -0.90283203125, -0.91943359375, -0.92578125, -0.8798828125, -0.94873046875, -0.6513671875, -0.10699462890625, -0.251220703125, -0.2410888671875, -0.2232666015625, -0.62841796875, -0.73828125, -0.68115234375, -0.748046875, -0.72265625, -0.74560546875, -0.8017578125, -0.400146484375, -0.372314453125, -0.4033203125, -0.52197265625, -0.471923828125, -0.8896484375, -0.7734375, -0.430908203125, -0.51416015625, -0.328369140625, -0.53857421875, -0.54541015625, -0.58154296875, -0.5859375, -0.529296875, -0.8115234375]}
# in jsonl

# need to look at four sources
# data_json at /home/pyp/vqhighlight/data/highlight_val_release_exist.jsonl
# clip sentence embedding in /saltpool0/data/pyp/vqhighlight/features/clip_text_features/qid0.npz
# video segments at /saltpool0/data/pyp/vqhighlight/scenedetect20/_6hnl_BrFvs_60_210-Scenes.csv
# video segment clip embedding at /saltpool0/data/pyp/vqhighlight/scenedetect_features/scenedetect20/_6hnl_BrFvs_60_210.npz

# submission stored at /saltpool0/data/pyp/vqhighlight/submission/clip/scenedetect20/submission.jsonl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Easy video feature extractor')

    parser.add_argument("--scenedetect_folder", type=str, help="scenedetect20, scenedetect25, scenedetect27, scenedetect30, scenedetect32")
    parser.add_argument("--sliding_window", action="store_true", default=False)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--fixed_scenes", action="store_true", default=False)
    parser.add_argument("--watershed", action="store_true", default=False)
    parser.add_argument("--gammas", nargs="+", type=float, default=None)
    parser.add_argument("--taus", nargs="+", type=float, default=None)
    parser.add_argument("--iou_threshold", type=float, default=None)
    parser.add_argument("--drop_threshold", type=float, default=None, help="drop the predicted window if the similarity score is below the threshold, this is on unnormalized threshold")
    args = parser.parse_args()
    with jsonlines.open("/home/pyp/vqhighlight/data/highlight_val_release_exist.jsonl", "r") as data_json:
        if not os.path.isdir(os.path.join("/saltpool0/data/pyp/vqhighlight/submission/clip/", args.scenedetect_folder)):
            os.makedirs(os.path.join("/saltpool0/data/pyp/vqhighlight/submission/clip/", args.scenedetect_folder))
        with jsonlines.open(os.path.join("/saltpool0/data/pyp/vqhighlight/submission/clip/", args.scenedetect_folder, "submission.jsonl"), "w") as writer:
            gt_window_len = []
            pred_window_len = []
            iou = []
            total_sim = []
            results = []
            for item in data_json:
                cur_gt_window_len = [cur_w[1] - cur_w[0] for cur_w in item['relevant_windows']]
                gt_window_len += cur_gt_window_len
                # load query
                query_embed_fn = "/saltpool0/data/pyp/vqhighlight/features/clip_text_features/qid" + str(item['qid']) + ".npz"
                query_embed = np.load(query_embed_fn)
                
                # load segmentation
                vid = item['vid'] # exclude ".mkv"
                temp = vid.split("_")
                start, end = temp[-2], temp[-1]
                suffix = "_" + start + "_" + end
                ytvid = vid[:-len(suffix)]
                start, end = float(start), float(end)
                anuj_vid = ytvid + "_" + str(int(start)) + "_" + str(int(end))
                if args.sliding_window:
                    assert args.window_size != None and args.stride != None
                    video_timestamps = []
                    for i in range(0,int(end-start),args.stride):
                        start_time = i
                        end_time = i + args.window_size
                        video_timestamps.append((start_time, end_time))
                else:
                    vid_seg_fn = os.path.join("/saltpool0/data/pyp/vqhighlight/fixed_scenes", args.scenedetect_folder, anuj_vid + "-Scenes.csv") if args.fixed_scenes else os.path.join("/saltpool0/data/pyp/vqhighlight/", args.scenedetect_folder, anuj_vid + "-Scenes.csv")
                    with open(vid_seg_fn, newline='') as f:
                        next(f)
                        reader = csv.DictReader(f)
                        video_timestamps = []
                        for row in reader:
                            start_time = float(row['Start Time (seconds)'])
                            end_time = float(row['End Time (seconds)'])
                            video_timestamps.append((start_time, end_time))
                        if len(video_timestamps) == 1:
                            video_timestamps = []
                            for i in range(0,int(end-start),10):
                                start_time = i
                                end_time = i + 15
                                video_timestamps.append((start_time, end_time))
                    
                # load video feature and calculate the similarity with query
                pred_relevant_windows = []
                # vid_feats_fn = "/saltpool0/data/pyp/vqhighlight/features/clip_features/" + vid + ".npz"
                vid_feats_fn = "/saltpool0/data/pyp/vqhighlight/scenedetect_features/uniform/" + anuj_vid + ".npz"
                if not os.path.isfile(vid_feats_fn):
                    vid_feats_fn = "/saltpool0/data/pyp/vqhighlight/scenedetect_features/uniform/" + vid + ".npz"
                    assert os.path.isfile(vid_feats_fn)
                vid_feats = dict(np.load(vid_feats_fn))['features']
                pooler_output_norm = LA.norm(query_embed['pooler_output'])
                cur_sim = []
                record_max = None
                record_window = None
                for start_time, end_time in video_timestamps:
                    if args.sliding_window:
                        feat_id_in_this_seg = list(range(int(np.floor(start_time/2)), min(int(np.ceil(end_time/2)), len(vid_feats))))
                    else:
                        feat_id_in_this_seg = list(range(int(np.floor(start_time)), min(int(np.ceil(end_time)), len(vid_feats))))
                    # print(feat_id_in_this_seg)
                    vid_feat = vid_feats[feat_id_in_this_seg,:]
                    if len(vid_feat.shape) == 1:
                        vid_feat = vid_feat[None,:]
                    # print(vid_feat.shape)
                    # print(query_embed['pooler_output'].shape)
                    sim = []
                    for cur_vid_feat in vid_feat:
                        dot_prod = sum(cur_vid_feat*query_embed['pooler_output']/(LA.norm(cur_vid_feat)*pooler_output_norm))
                        sim.append(dot_prod)
                        assert dot_prod <= 1.1
                    if len(sim) > 0:
                        max_sim = np.max(sim).astype(np.float64)
                        if record_max == None or record_max <= max_sim:
                            record_max = max_sim
                            record_window = [start_time,end_time,max_sim]
                        if args.drop_threshold == None:
                            cur_sim.append(max_sim)
                            pred_relevant_windows.append([start_time,end_time,max_sim])
                        elif max_sim >= args.drop_threshold:
                            cur_sim.append(max_sim)
                            pred_relevant_windows.append([start_time,end_time,max_sim])
                # if the drop_threshold filtered out all the prediction, keep the prediction with the max similarity 
                if len(pred_relevant_windows) == 0:
                    cur_sim.append(record_max)
                    pred_relevant_windows.append(record_window)
                temp = pred_relevant_windows[np.argmax(cur_sim)]
                gt_s = item['relevant_windows'][0][0]
                gt_e = item['relevant_windows'][0][1]
                p_s = temp[0]
                p_e = temp[1]
                intersec = max(0, min(p_e, gt_e) - max(p_s, gt_s))
                union = min(p_e - p_s + gt_e - gt_s, max(p_e, gt_e) - min(p_s, gt_s))
                iou.append(intersec/union)
                pred_window_len.append(temp[1] - temp[0])
                # scale the similarity score to 0~1
                cur_sim = np.array(cur_sim)
                # do not normalize the score
                # if np.max(cur_sim) - np.min(cur_sim) > 0. :
                #     cur_sim = (cur_sim - np.min(cur_sim))/(np.max(cur_sim) - np.min(cur_sim))
                # else:
                #     print("Not here")
                pred_relevant_windows = [[item[0], item[1], sim] for item, sim in zip(pred_relevant_windows,cur_sim)]
                pred_relevant_windows.sort(key = lambda x: x[2], reverse = True)
                # pred_relevant_windows = [item for item in pred_relevant_windows if item[1] - item[0] >= 4 and item[1] - item[0] <= 60]
                # print(pred_relevant_windows)
                # if len(pred_relevant_windows) > 1:
                results.append({
                    "qid": item['qid'],
                    "query": item['query'],
                    "vid": item['vid'],
                    "pred_relevant_windows": pred_relevant_windows
                })
                    # total_sim += cur_sim
            if args.watershed:
                assert args.gammas != None and args.taus != None and args.iou_threshold != None
                import joblib
                parallizer = joblib.Parallel(n_jobs=64, max_nbytes=None, verbose=2)
                watershed_windows_out = parallizer(joblib.delayed(watershed)(item['pred_relevant_windows'], args.gammas, args.taus, args.iou_threshold) for item in results)
                results = [{'qid': item['qid'], 'query': item['query'], 'vid': item['vid'], "pred_relevant_windows": watershed_windows if len(watershed_windows) > 0 else item['pred_relevant_windows']} for watershed_windows, item in zip(watershed_windows_out, results)]
            for item in results:
                writer.write({
                            "qid": item['qid'],
                            "query": item['query'],
                            "vid": item['vid'],
                            "pred_relevant_windows": item['pred_relevant_windows']
                        })
                # print(pred_relevant_windows)
                # assert False
            # plt.figure()
            # sns.histplot(total_sim)
            # plt.savefig(f"./similarity_dist_{args.scenedetect_folder}.png")
            # plt.figure()
            # sns.histplot(pred_window_len)
            # plt.savefig(f"./pred_window_len_{args.scenedetect_folder}.png")
            # plt.figure()
            # sns.histplot(iou)
            # plt.savefig(f"./iou_{args.scenedetect_folder}.png")
    # with jsonlines.open("/home/pyp/vqhighlight/data/highlight_val_release_exist.jsonl", "r") as data_json:
    #     window_len = []
    #     for item in data_json:
    #         window_len += [cur_w[1] - cur_w[0] for cur_w in item['relevant_windows']]
                
    #     sns.histplot(window_len)
    #     plt.savefig("./ground_truth_window_len.png")    

