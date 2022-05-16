import numpy as np

def _union(windows):
    """
    modifed from https://stackoverflow.com/questions/15273693/union-of-multiple-ranges

    return a list of windows that is the union of the input
    
    input windows: [[7.1, 11.5, [0.32]], [11.0, 13.0, [0.41]], [11.0, 14.9, [0.32]], [15.0, 20.0, [0.27, 0.35]], [23.0, 39.0, [0.40, 0.44]]]
    input doesn't need to be sorted

    return: [[7.1, 14.9, [0.32, 0.41, 0.32]], [15.0, 20.0, [0.27, 0.35]], [23., 39., [0.40, 0.44]]]
    the similarity score is just the mean of combined segments
    """
    b = []
    for begin, end, score in sorted(windows):
        if b and b[-1][1] >= begin:
            b[-1][1] = max(b[-1][1], end)
            b[-1][2] += score
        else:
            b.append([begin, end, score])
    return b

def temporal_nms(bboxes, thresh, score_ind=2):
    """
    One-dimensional non-maximal suppression
    :param bboxes: [[st, ed, score], ...]
    :param thresh:
    :return:


    copied (and slightly modified) from https://github.com/yjxiong/action-detection/blob/3d8ff836387fb977979ef792152a6b6ac43725fc/ops/sequence_funcs.py#L71
    """
    t1 = np.array([x[0] for x in bboxes])
    t2 = np.array([x[1] for x in bboxes])
    scores = np.array([x[score_ind] for x in bboxes])

    durations = t2 - t1 + 1
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        tt1 = np.maximum(t1[i], t1[order[1:]])
        tt2 = np.minimum(t2[i], t2[order[1:]])
        intersection = tt2 - tt1 + 1
        IoU = intersection / (durations[i] + durations[order[1:]] - intersection).astype(float)

        inds = np.where(IoU <= thresh)[0]
        order = order[inds + 1]

    return [bboxes[i] for i in keep]


def watershed_single(windows, gamma, tau):
    """
    run 1D watershed algo on predicted windows
    method introduced in https://arxiv.org/pdf/1704.06228.pdf
    
    input:
        gamma: similarity threshold
        tau: window_dur/total_dur threshold
        windows: [[1.2, 15.3, 0.32], [15.3, 18.6, 0.34], [18.6, 23.7, 0.12], [23.7, 24.6, 0.44], [25.7, 28.4, 0.32], [30.1, 34.5, 0.45], [34.5, 35.6, 0.01], [35.6, 38.4, 0.34]]
            segments of [start_time, end_time, similarity_score]
            the windows can overlap
            windows should be sorted based on start_time

        tau and gamma are floating point between 0 and 1
    
    output:
    (suppose gamma = 0.32 and tau = 0.85)
        merged_windows: [[1.2, 18.6, [0.32, 0.34]],
                    [23.7, 24.6, [0.44]],
                    [25.7, 28.4, [0.32]],
                    [30.1, 38.4, [0.45, 0.34]]]
        merged segments
    """
    scores = [item[2] for item in windows]
    remain_windows = np.array(windows)[np.where(np.array(scores) >= gamma)[0]]
    if len(remain_windows) == 0:
        return None
    remain_windows = [[w[0], w[1], [w[2]]] for w in remain_windows] # [[1.2, 15.3, 0.32], [15.3, 18.6, 0.34], ....] -> [[1.2, 15.3, [0.32]], [15.3, 18.6, [0.34]], ....] 
    groups = []
    for i, cur_win in enumerate(remain_windows):
        window_union = _union(groups[-1]+[cur_win]) if groups else [cur_win]
        window_duration = sum([item[1] - item[0] for item in window_union])
        total_duration = window_union[-1][1] - window_union[0][0]
        if window_duration/total_duration <= tau or i == 0:
            groups.append([cur_win])
        else:
            groups[-1] = window_union
        merged_groups = [[cur_group[0][0],cur_group[-1][1], [ss for s in cur_group for ss in s[2]]] for cur_group in groups]
    return merged_groups


def watershed(windows, gammas, taus, iou_threshold=0.95):
    """
    step1: run the above watershed algo with different gamma and tau  
    step2: the union is taken on generated sets of proposals
    step3: then perform non-maximal suppression with given iou_threshold

    method introduced in https://arxiv.org/pdf/1704.06228.pdf

    output format:
        [[23.7, 24.6, 0.44], [30.1, 38.4, 0.41], [1.2, 18.6, 0.33], [25.7, 28.4, 0.32]]
        (sorted by similarity scores in an decreasing order)
    """
    # step 1
    all_proposals = []
    for gamma in gammas:
        for tau in taus:
            cur_proposal =  watershed_single(windows, gamma, tau)
            if cur_proposal != None:
                all_proposals += cur_proposal
    # step 2
    proposals = [[item[0], item[1], np.mean(item[2])] for item in _union(all_proposals)]
    
    # step 3
    proposals = temporal_nms(proposals, thresh = iou_threshold)
    return proposals




            