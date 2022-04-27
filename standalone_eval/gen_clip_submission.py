import os
import numpy as np
from numpy import linalg as LA
import jsonlines
import argparse
import csv
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
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
    args = parser.parse_args()
    with jsonlines.open("/home/pyp/vqhighlight/data/highlight_val_release_exist.jsonl", "r") as data_json:
        if not os.path.isdir(os.path.join("/saltpool0/data/pyp/vqhighlight/submission/clip/", args.scenedetect_folder)):
            os.makedirs(os.path.join("/saltpool0/data/pyp/vqhighlight/submission/clip/", args.scenedetect_folder))
        with jsonlines.open(os.path.join("/saltpool0/data/pyp/vqhighlight/submission/clip/", args.scenedetect_folder, "submission.jsonl"), "w") as writer:
            gt_window_len = []
            pred_window_len = []
            iou = []
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
                vid_seg_fn = os.path.join("/saltpool0/data/pyp/vqhighlight/", args.scenedetect_folder, anuj_vid + "-Scenes.csv")
                if args.sliding_window:
                    video_timestamps = []
                    for i in range(0,int(end-start),10):
                        start_time = i
                        end_time = i + 15
                        video_timestamps.append((start_time, end_time))
                else:
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
                vid_feats_fn = "/saltpool0/data/pyp/vqhighlight/features/clip_features/" + vid + ".npz"
                vid_feats = dict(np.load(vid_feats_fn))['features']
                pooler_output_norm = LA.norm(query_embed['pooler_output'])
                cur_sim = []
                for start_time, end_time in video_timestamps:
                    feat_id_in_this_seg = list(range(int(np.floor(start_time/2)), min(int(np.ceil(end_time/2)), len(vid_feats))))
                    vid_feat = vid_feats[feat_id_in_this_seg,:]
                    if len(vid_feat.shape) == 1:
                        vid_feat = vid_feat[None,:]
                    # print(vid_feat.shape)
                    # print(query_embed['pooler_output'].shape)
                    sim = []
                    for cur_vid_feat in vid_feat:
                        dot_prod = sum(cur_vid_feat*query_embed['pooler_output']/(LA.norm(cur_vid_feat)*pooler_output_norm))
                        sim.append(dot_prod)
                    if len(sim) > 0:
                        cur_sim.append(np.max(sim).astype(np.float64))
                        pred_relevant_windows.append([start_time,end_time,np.max(sim).astype(np.float64)])
                temp = pred_relevant_windows[np.argmax(cur_sim)]
                gt_s = item['relevant_windows'][0][0]
                gt_e = item['relevant_windows'][0][1]
                p_s = temp[0]
                p_e = temp[1]
                intersec = max(0, min(p_e, gt_e) - max(p_s, gt_s))
                union = min(p_e - p_s + gt_e - gt_s, max(p_e, gt_e) - min(p_s, gt_s))
                iou.append(intersec/union)
                pred_window_len.append(temp[1] - temp[0])
                # pred_relevant_windows = []
                # cur_sim = []
                # vid_feats_fn = "/saltpool0/data/pyp/vqhighlight/features/clip_features/" + vid + ".npz"
                # vid_feats = dict(np.load(vid_feats_fn))['features']
                # pooler_output_norm = LA.norm(query_embed['pooler_output'])
                # for i, feat in enumerate(vid_feats):
                #     start_time, end_time = i*2, (i+1)*2
                #     dot_prod = sum(feat*query_embed['pooler_output']/(LA.norm(feat)*pooler_output_norm))
                #     cur_sim.append(dot_prod)
                #     pred_relevant_windows.append([start_time,end_time,dot_prod.astype(np.float64)])
                    # print(sim)
                        # assert False
                # vid_feats_fn = "/saltpool0/data/pyp/vqhighlight/scenedetect_features/" + args.scenedetect_folder + "/" + anuj_vid + ".npz"
                # vid_feats = dict(np.load(vid_feats_fn))
                # cur_sim = []
                # if len(vid_feats) == len(video_timestamps):
                #     for vid_feat_key, (start_time, end_time) in zip(vid_feats, video_timestamps):
                #         vid_feat = vid_feats[vid_feat_key]
                #         if len(vid_feat.shape) == 1:
                #             vid_feat = vid_feat[None,:]
                #         # print(vid_feat.shape)
                #         # print(query_embed['pooler_output'].shape)
                #         sim = []
                #         pooler_output_norm = LA.norm(query_embed['pooler_output'])
                #         for cur_vid_feat in vid_feat:
                #             dot_prod = sum(cur_vid_feat*query_embed['pooler_output']/(LA.norm(cur_vid_feat)*pooler_output_norm))
                #             sim.append(dot_prod)
                #         # print(sim)
                #         if len(sim) > 0:
                #             cur_sim.append(np.max(sim).astype(np.float64))
                #             pred_relevant_windows.append([start_time,end_time,np.max(sim).astype(np.float64)])
                #         # assert False
                # else:
                #     assert False 
                #     print("video timesteps:", len(video_timestamps))
                #     print(len(vid_feats))
                #     print("feature length: ", vid_feats['0'].shape)
                #     for start_time, end_time in video_timestamps:
                #         for cur_vid_feat in vid_feat:
                #             sim.append(vid_feats['0'][start_time:end_time]*query_embed['pooler_output']/(LA.norm(vid_feats['0'][start_time:end_time])*LA.norm(query_embed['pooler_output'])))
                #         pred_relevant_windows.append([start_time,end_time,np.max(sim)])
                
                # print(pred_relevant_windows)
                if len(pred_relevant_windows) > 0:
                    # cur_sim = np.array(cur_sim) / max(cur_sim)
                    # print(cur_sim)
                    if len(pred_relevant_windows) == 1:
                        print(pred_relevant_windows)
                    # for i in range(len(pred_relevant_windows)):
                    #     pred_relevant_windows[i][2] = 0.
                    writer.write({
                        "qid": item['qid'],
                        "query": item['query'],
                        "vid": item['vid'],
                        "pred_relevant_windows": pred_relevant_windows
                    })

                # print(pred_relevant_windows)
                # assert False
            plt.figure()
            sns.histplot(pred_window_len)
            plt.savefig(f"./pred_window_len_{args.scenedetect_folder}.png")
            plt.figure()
            sns.histplot(iou)
            plt.savefig(f"./iou_{args.scenedetect_folder}.png")
    # with jsonlines.open("/home/pyp/vqhighlight/data/highlight_val_release_exist.jsonl", "r") as data_json:
    #     window_len = []
    #     for item in data_json:
    #         window_len += [cur_w[1] - cur_w[0] for cur_w in item['relevant_windows']]
                
    #     sns.histplot(window_len)
    #     plt.savefig("./ground_truth_window_len.png")    

