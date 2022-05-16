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
import networkx as nx
import numpy as np
from networkx.algorithms.bipartite.matrix import from_biadjacency_matrix
from scipy.sparse import csr_matrix
import torch
import collections
# the goal is to generate the following format
# {"qid": 2579, "query": "A girl and her mother cooked while talking with each other on facetime.", "vid": "NUsG9BgSes0_210.0_360.0", "pred_relevant_windows": [[0.0, 70.0, 0.9986], [78.0, 146.0, 0.4138], [0.0, 146.0, 0.0444], [108.0, 150.0, 0.0364], [0.0, 48.0, 0.0081], [56.0, 116.0, 0.0071], [38.0, 82.0, 0.0006], [62.0, 88.0, 0.0006], [132.0, 146.0, 0.0001], [52.0, 88.0, 0.0]], "pred_saliency_scores": [-0.2452392578125, -0.3779296875, -0.474609375, -0.474365234375, -0.52880859375, -0.267578125, -0.27197265625, -0.19580078125, -0.2034912109375, -0.058135986328125, -0.2442626953125, -0.2142333984375, -0.02569580078125, -0.09539794921875, 0.0751953125, 0.073486328125, -0.009185791015625, 0.028411865234375, 0.060882568359375, 0.1505126953125, 0.1416015625, -0.04473876953125, 0.02642822265625, -0.041900634765625, 0.0595703125, 0.127685546875, 0.10394287109375, 0.22216796875, 0.07666015625, 0.02508544921875, 0.1064453125, 0.1470947265625, 0.11492919921875, 0.1197509765625, 0.1614990234375, 0.12890625, 0.0863037109375, -0.0022640228271484375, 0.1641845703125, 0.16259765625, -0.02667236328125, -0.90771484375, -0.90283203125, -0.91943359375, -0.92578125, -0.8798828125, -0.94873046875, -0.6513671875, -0.10699462890625, -0.251220703125, -0.2410888671875, -0.2232666015625, -0.62841796875, -0.73828125, -0.68115234375, -0.748046875, -0.72265625, -0.74560546875, -0.8017578125, -0.400146484375, -0.372314453125, -0.4033203125, -0.52197265625, -0.471923828125, -0.8896484375, -0.7734375, -0.430908203125, -0.51416015625, -0.328369140625, -0.53857421875, -0.54541015625, -0.58154296875, -0.5859375, -0.529296875, -0.8115234375]}
# in jsonl

# need to look at four sources
# data_json at /home/pyp/vqhighlight/data/highlight_val_release_exist.jsonl
# clip sentence embedding in /saltpool0/data/pyp/vqhighlight/features/clip_text_features/qid0.npz
# video segments at /saltpool0/data/pyp/vqhighlight/scenedetect20/_6hnl_BrFvs_60_210-Scenes.csv
# video segment clip embedding at /saltpool0/data/pyp/vqhighlight/scenedetect_features/scenedetect20/_6hnl_BrFvs_60_210.npz

# submission stored at /saltpool0/data/pyp/vqhighlight/submission/clip/scenedetect20/submission.jsonl
def _permute(edge, sim_mat):
    # Edge not in l,r order. Fix it
    if edge[0] < sim_mat.shape[0]:
        return edge[0], edge[1] - sim_mat.shape[0]
    else:
        return edge[1], edge[0] - sim_mat.shape[0]

def get_align(x, y):
    # both x and y are list of seg boundaries of format [[0.32, 0.40], [0.44, 0.46], ...]
    n = len(x) 
    m = len(y)
    sim_mat = 1000 * np.ones((n, m))
    
    # retrieve center of each gt and Jason's word segments
    # x_center = [np.mean([x[1], x[2]]) for x in x]
    x_center = np.mean(x, axis=1)
    y_center = np.mean(y, axis=1)
    
    # compute l1_distance
    for i in range(n): 
        for j in range(m): 
            # sim_mat[i,j] = - np.abs(x_center[i] - y_center[j])
            intersection = max(0, min(x[i][1], y[j][1]) - max(x[i][0], y[j][0]))
            union = 1. if intersection == 0. else max(x[i][1], y[j][1]) - min(x[i][0], y[j][0])
            sim_mat[i,j] = intersection/union
    G = from_biadjacency_matrix(csr_matrix(sim_mat))
    matching = nx.max_weight_matching(G, maxcardinality=True)
    alignment = [_permute(x, sim_mat) for x in matching]
    best_ids = [item[0] for item in alignment]
    best_scores = [sim_mat[i,j] for i, j in alignment]
    scores, inds = torch.from_numpy(sim_mat).topk(n, 0) # [k, 3]
    # print(best_scores)
    ind_score = collections.OrderedDict()
    for k, (score, ind) in enumerate(zip(scores.view(-1), inds.view(-1))):
        if ind not in best_ids and score > 0:
            if ind in ind_score:
                if score > ind_score[ind]:
                    ind_score[ind.item()] = score.item()
            else:
                ind_score[ind.item()] = score.item()
    # print(ind_score)
    secondary_inds = sorted(ind_score, key=lambda x: ind_score[x], reverse=True)

    return [[alignment[i][0], best_scores[i]] for i in range(len(alignment))], [[secondary_ind, ind_score[secondary_ind]] for secondary_ind in secondary_inds]

def get_combined_align(x, y):
    # x: predicted unit segments
    # y: GT segments

    # combine the predicted units into larger segments so that they have most IoU with GT segments
    assignment_s = []
    assignment_e = []
    b_assignment_s = []
    b_assignment_e = []
    b_candidate = []
    new_x = []
    iou = []
    used = []
    # print("x", x)
    # print("y", y)
    for i, (gt_s, gt_e) in enumerate(y):
        assignment_s.append([])
        assignment_e.append([])
        b_assignment_s.append([])
        b_assignment_e.append([])
        b_candidate.append([])
        for j, (pred_s, pred_e) in enumerate(x):
            if pred_s >= gt_s and pred_e <= gt_e:
                assignment_s[i].append(pred_s)
                assignment_e[i].append(pred_e)
                used.append(j)
            elif pred_s <= gt_s and pred_e >= gt_e and j not in used:
                assignment_s[i].append(pred_s)
                assignment_e[i].append(pred_e)
                used.append(j)
                break
            elif pred_s <= gt_s and pred_e <= gt_e and pred_e >= gt_s: # pred_s  gt_s     pred_e   gt_e
                if pred_e - gt_s >= gt_s - pred_s and j not in used:
                    assignment_s[i].append(pred_s)
                    assignment_e[i].append(pred_e)
                    used.append(j)
                else:
                    b_assignment_s[i].append(pred_s)
                    b_assignment_e[i].append(pred_e)
                    b_candidate[i].append(j)
            elif pred_s >= gt_s and pred_s <= gt_e and pred_e >= gt_e: # gt_s   pred_s  gt_e pred_e
                if gt_e - pred_s >= pred_e - gt_e and j not in used:
                    assignment_s[i].append(pred_s)
                    assignment_e[i].append(pred_e)
                    used.append(j)
                else:
                    b_assignment_s[i].append(pred_s)
                    b_assignment_e[i].append(pred_e)
                    b_candidate[i].append(j)
        if len(assignment_e[i]) == 0:
            cur_iou = 0
            new_s, new_e = 0,0
            cur_j = -1
            flag = False
            for j, temp_new_s, temp_new_e in zip(b_candidate[i], b_assignment_s[i], b_assignment_e[i]):
                intersection = max(0, min(temp_new_e, gt_e) - max(temp_new_s, gt_s))
                union = max(temp_new_e, gt_e) - min(temp_new_s, gt_s)
                temp = intersection/union
                if temp >= cur_iou and j not in used:
                    cur_iou = temp
                    new_s = temp_new_s
                    new_e = temp_new_e
                    cur_j = j
            if len(b_assignment_e[i]) == 2:
                temp_new_s, temp_new_e = min(b_assignment_s[i]), max(b_assignment_e[i])
                intersection = max(0, min(temp_new_e, gt_e) - max(temp_new_s, gt_s))
                union = max(temp_new_e, gt_e) - min(temp_new_s, gt_s)
                temp = intersection/union
                if temp > cur_iou and j not in used:
                    cur_iou = temp
                    new_s = temp_new_s
                    new_e = temp_new_e
                    used.extend(b_candidate[i])
                    flag = True
            if not flag and cur_j != -1:
                used.append(cur_j)
        else:
            new_s, new_e = min(assignment_s[i]), max(assignment_e[i])
            intersection = max(0, min(new_e, gt_e) - max(new_s, gt_s))
            union = max(new_e, gt_e) - min(new_s, gt_s)
            cur_iou = intersection/union
        if cur_iou == 0 and new_s !=0 and new_e != 0:
            print("gt:", gt_s, gt_e)
            print("new_s, new_e", new_s, new_e)
            print("assignment:", assignment_s[i], assignment_e[i])
            print("b_assignment: ", b_assignment_s[i], b_assignment_e[i])
        new_x.append([new_s, new_e])
        iou.append(cur_iou)
    # print("new_x", new_x)
    # print("gt", y)
    assert len(new_x) == len(y)
    return [[s, e, score] for (s, e), score in zip(new_x, iou)]
        # 
    

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
    parser.add_argument("--combine_units", action="store_true", default=False)
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
                # query_embed_fn = "/saltpool0/data/pyp/vqhighlight/features/clip_text_features/qid" + str(item['qid']) + ".npz"
                # query_embed = np.load(query_embed_fn)
                
                # load segmentation
                vid = item['vid'] # exclude ".mkv"
                temp = vid.split("_")
                start, end = temp[-2], temp[-1]
                suffix = "_" + start + "_" + end
                ytvid = vid[:-len(suffix)]
                start, end = float(start), float(end)
                anuj_vid = ytvid + "_" + str(int(start)) + "_" + str(int(end))
                pred_relevant_windows = []
                video_timestamps = []
                if args.sliding_window:
                    assert args.window_size != None and args.stride != None
                    for i in range(0,int(end-start),args.stride):
                        start_time = i
                        end_time = i + args.window_size
                        video_timestamps.append([start_time, end_time])
                else:
                    vid_seg_fn = os.path.join("/saltpool0/data/pyp/vqhighlight/fixed_scenes", args.scenedetect_folder, anuj_vid + "-Scenes.csv") if args.fixed_scenes else os.path.join("/saltpool0/data/pyp/vqhighlight/", args.scenedetect_folder, anuj_vid + "-Scenes.csv")
                    with open(vid_seg_fn, newline='') as f:
                        next(f)
                        reader = csv.DictReader(f)
                        for row in reader:
                            start_time = float(row['Start Time (seconds)'])
                            end_time = float(row['End Time (seconds)'])
                            video_timestamps.append([start_time, end_time])
                        if len(video_timestamps) == 1:
                            video_timestamps = []
                            for i in range(0,int(end-start),10):
                                start_time = i
                                end_time = i + 15
                                video_timestamps.append([start_time, end_time])
                if args.combine_units:
                    pred_relevant_windows = get_combined_align(video_timestamps, item['relevant_windows'])
                else:
                    best_inds_scores, secondary_inds_scores = get_align(video_timestamps, item['relevant_windows'])
                    pred_relevant_windows = [[video_timestamps[ind][0], video_timestamps[ind][1], score] for ind, score in best_inds_scores]
                    pred_relevant_windows.extend([[video_timestamps[ind][0], video_timestamps[ind][1], score] for ind, score in secondary_inds_scores])
                pred_relevant_windows.sort(key = lambda x: x[2], reverse = True)
                # if len(pred_relevant_windows) > 1:
                writer.write({
                        "qid": item['qid'],
                        "query": item['query'],
                        "vid": item['vid'],
                        "pred_relevant_windows": pred_relevant_windows
                    })
                    # results.append({
                    #     "qid": item['qid'],
                    #     "query": item['query'],
                    #     "vid": item['vid'],
                    #     "pred_relevant_windows": pred_relevant_windows
                    # })
                    # total_sim += cur_sim
            # if args.watershed:
            #     assert args.gammas != None and args.taus != None and args.iou_threshold != None
            #     import joblib
            #     parallizer = joblib.Parallel(n_jobs=64, max_nbytes=None, verbose=2)
            #     watershed_windows_out = parallizer(joblib.delayed(watershed)(item['pred_relevant_windows'], args.gammas, args.taus, args.iou_threshold) for item in results)
            #     results = [{'qid': item['qid'], 'query': item['query'], 'vid': item['vid'], "pred_relevant_windows": watershed_windows if len(watershed_windows) > 0 else item['pred_relevant_windows']} for watershed_windows, item in zip(watershed_windows_out, results)]
            # for item in results:
            #     writer.write({
            #                 "qid": item['qid'],
            #                 "query": item['query'],
            #                 "vid": item['vid'],
            #                 "pred_relevant_windows": item['pred_relevant_windows']
            #             })
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

