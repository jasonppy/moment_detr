"""
https://github.com/yjxiong/action-detection

"""
import numpy as np
from scipy.ndimage import gaussian_filter


def temporal_nms(bboxes, thresh):
    """One-dimensional non-maximal suppression
    Args:
        bboxes: np.ndarray of shape (n_props, 3), each row is [st, ed, score]
        thresh: float

    Returns:

    """
    if len(bboxes) == 1:
        return bboxes
    t1 = bboxes[:, 0]
    t2 = bboxes[:, 1]
    scores = bboxes[:, 2]

    durations = t2 - t1
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        tt1 = np.maximum(t1[i], t1[order[1:]])
        tt2 = np.minimum(t2[i], t2[order[1:]])
        intersection = tt2 - tt1
        iou = intersection / (durations[i] + durations[order[1:]] - intersection).astype(float)

        inds = np.where(iou <= thresh)[0]
        order = order[inds + 1]
    return bboxes[keep, :]


def softmax(raw_score, temperature=1):
    """works with any dimension"""
    exp_s = np.exp((raw_score - raw_score.max(axis=-1)[..., None])*temperature)
    sum_s = exp_s.sum(axis=-1)
    return exp_s / sum_s[..., None]


def label_frame_by_threshold_raw(frm_scores, bw=None, thresh=list([0.05])):
    """Build frame labels by thresholding the foreground class responses
    Args:
        frm_scores: np.ndarray (n_frm), 1D
        bw: int, smooth the frm_scores with a 1D gaussian_filter
        thresh: list(float)
    """
    softmax_frm_scores = softmax(frm_scores)

    rst = []
    cls_score = softmax_frm_scores if bw is None else gaussian_filter(softmax_frm_scores, bw)
    for th in thresh:
        rst.append((cls_score > th, frm_scores))
    return rst


def label_frame_by_threshold(frm_scores, bw=None, thresh=list([0.05])):
    """Build frame labels by thresholding the foreground class responses
    Args:
        frm_scores: np.ndarray (n_frm), 1D, assume sigmoid scores
        bw: int, smooth the frm_scores with a 1D gaussian_filter
        thresh: list(float)
    """
    # softmax_frm_scores = softmax(frm_scores)

    rst = []
    cls_score = frm_scores if bw is None else gaussian_filter(frm_scores, bw)
    for th in thresh:
        rst.append((cls_score > th, frm_scores))
    return rst


def build_box_by_search(frm_label_lst, tol, reduction="mean"):
    """
    Args:
        frm_label_lst: list(tuple),
            each tuple is (cls_score > th (1D np.ndarray), frm_scores (1D np.ndarray))
        tol: list(float)
        reduction: str, choices=["mean", "sum"], indicate how the proposal scores are calculated,
            whether take mean or summation of all clip-level scores inside the proposal region.
            The original TAG method uses sum, which always favor longer proposals, see
            https://github.com/yjxiong/action-detection/issues/68.

    Returns:

    """
    reduction_func = np.mean if reduction == "mean" else np.sum
    boxes = []
    for frm_labels, frm_scores in frm_label_lst:
        length = len(frm_labels)
        diff = np.empty(length+1)
        diff[1:-1] = frm_labels[1:].astype(int) - frm_labels[:-1].astype(int)
        diff[0] = float(frm_labels[0])
        diff[length] = 0 - float(frm_labels[-1])
        cs = np.cumsum(1 - frm_labels)
        offset = np.arange(0, length, 1)

        up = np.nonzero(diff == 1)[0]  # up edges
        down = np.nonzero(diff == -1)[0]  # down edges

        assert len(up) == len(down), "{} != {}".format(len(up), len(down))
        for i, t in enumerate(tol):
            signal = cs - t * offset
            for x in range(len(up)):
                s = signal[up[x]]
                for y in range(x + 1, len(up)):
                    if y < len(down) and signal[up[y]] > s:
                        boxes.append((up[x], down[y-1]+1, reduction_func(frm_scores[up[x]:down[y-1]+1])))
                        break
                else:
                    boxes.append((up[x], down[-1] + 1, reduction_func(frm_scores[up[x]:down[-1] + 1])))

            for x in range(len(down) - 1, -1, -1):
                s = signal[down[x]] if down[x] < length else signal[-1] - t
                for y in range(x - 1, -1, -1):
                    if y >= 0 and signal[down[y]] < s:
                        boxes.append((up[y+1], down[x] + 1, reduction_func(frm_scores[up[y+1]:down[x] + 1])))
                        break
                else:
                    boxes.append((up[0], down[x] + 1, reduction_func(frm_scores[0:down[x]+1 + 1])))
    return boxes


def generate_tag_proposals_single_video(scores, min_len=2, max_len=16, bw=3, clip_length=1.0):
    """
    Args:
        scores: np.ndarray (n_frm), 1d vector, range [0, 1] after sigmoid
        min_len: int, minimum length of predicted span
        max_len: int, maximum length of predicted span
        bw: gaussian_filter bw
        clip_length:
    """
    # print("raw scores", ",".join([str(e) for e in scores.tolist()]))
    # scores = scale_to_range01(scores)
    # print("scaled scores", ",".join([str(e) for e in scores.tolist()]))
    thresholds = [0.01, 0.05, 0.1, .15, 0.25, .4, .5, .6, .7, .8, .9, .95]
    topk_labels = label_frame_by_threshold(scores, bw=bw, thresh=thresholds)
    # print("topk_labels {}, {}".format(len(topk_labels), topk_labels[0]))

    bboxes = []
    tol_lst = [0.05, .1, .2, .3, .4, .5, .6, 0.8, 1.0]
    # tol_lst = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.12, 0.15, .2, .3, .4, .5, .6, 0.8, 1.0]
    bboxes.extend(build_box_by_search(topk_labels, np.array(tol_lst)))

    bboxes = np.array(bboxes)

    # print("bboxes before nms {}".format(len(bboxes)))
    if len(bboxes) > 1:
        bboxes = temporal_nms(bboxes, 0.99)
    # print("bboxes after nms {}".format(len(bboxes)))

    # filter out too short or too long proposals
    bboxes = list(filter(lambda b: max_len >= b[1] - b[0] >= min_len, bboxes))
    # print("bboxes after filter {}".format(len(bboxes)))
    bboxes = np.array(bboxes)
    bboxes[:, 1] += 1
    bboxes[:2] *= clip_length

    # each tuple is [st_frm_idx, ed_frm_idx, score], sort by scores
    pr_box = sorted(bboxes, key=lambda b: b[2], reverse=True)  # decreasing order
    # print("pr_box", pr_box[:3])
    # raise NotImplementedError
    return pr_box


def find_unique_rows_with_lexsort(data):
    """ Find unique rows indices
    https://stackoverflow.com/questions/31097247/remove-duplicate-rows-of-a-numpy-array
    """
    sorted_data = data[np.lexsort(data.T), :]
    row_mask = np.append([True], np.any(np.diff(sorted_data, axis=0), 1))
    return sorted_data[row_mask]


def build_box_by_search_with_id(frm_label_lst, tol, reduction="mean"):
    """
    Args:
        frm_label_lst: list(tuple),
            each tuple is (cls_score > th (2D np.ndarray), frm_scores (2D np.ndarray))
        tol: list(float)
        reduction: str, choices=["mean", "sum"], indicate how the proposal scores are calculated,
            whether take mean or summation of all clip-level scores inside the proposal region.
            The original TAG method uses sum, which always favor longer proposals, see
            https://github.com/yjxiong/action-detection/issues/68.

    Returns:

    """
    reduction_func = np.mean if reduction == "mean" else np.sum
    boxes = []
    for idx, frm_labels, frm_scores in frm_label_lst:
        length = len(frm_labels)
        diff = np.empty(length+1)
        diff[1:-1] = frm_labels[1:].astype(int) - frm_labels[:-1].astype(int)
        diff[0] = float(frm_labels[0])
        diff[length] = 0 - float(frm_labels[-1])
        cs = np.cumsum(1 - frm_labels)
        offset = np.arange(0, length, 1)

        up = np.nonzero(diff == 1)[0]  # up edges
        down = np.nonzero(diff == -1)[0]  # down edges

        assert len(up) == len(down), "{} != {}".format(len(up), len(down))
        for i, t in enumerate(tol):
            signal = cs - t * offset
            for x in range(len(up)):
                s = signal[up[x]]
                for y in range(x + 1, len(up)):
                    if y < len(down) and signal[up[y]] > s:
                        boxes.append((idx, up[x], down[y-1]+1, reduction_func(frm_scores[up[x]:down[y-1]+1])))
                        break
                else:
                    boxes.append((idx, up[x], down[-1] + 1, reduction_func(frm_scores[up[x]:down[-1] + 1])))

            for x in range(len(down) - 1, -1, -1):
                s = signal[down[x]] if down[x] < length else signal[-1] - t
                for y in range(x - 1, -1, -1):
                    if y >= 0 and signal[down[y]] < s:
                        boxes.append((idx, up[y+1], down[x] + 1, reduction_func(frm_scores[up[y+1]:down[x] + 1])))
                        break
                else:
                    boxes.append((idx, up[0], down[x] + 1, reduction_func(frm_scores[0:down[x]+1 + 1])))
    return boxes


def generate_tag_proposals_batched_video_single_q(scores, video_indices, video_scores, min_len=2, max_len=16,
                                                  bw=3, max_out=100, clip_length=1.):
    """a batch of videos for a single query
    Args:
        scores: np.ndarray (N, n_frm), 2d vector, range [0, 1] after sigmoid
        video_indices: (N, ) a list of unique video ids
        video_scores: (N, ) a list of scores,
        min_len: int, minimum length of predicted span
        max_len: int, maximum length of predicted span
        bw: gaussian_filter bw
        max_out: maximum number of output
        clip_length: float
    """
    assert np.max(scores) <= 1
    assert np.min(scores) >= 0
    # print("scores ", scores.shape, scores[:5, :5], scores[-5:, -5:])
    # print("raw scores", ",".join([str(e) for e in scores.tolist()]))
    # scores = scale_to_range01(scores)
    # print("scaled scores", ",".join([str(e) for e in scores.tolist()]))
    thresholds = np.array([0.01, 0.05, 0.1, .15, 0.25, .4, .5, .6, .7, .8, .9, .95])
    # topk_labels = label_frame_by_threshold(scores, bw=bw, thresh=thresholds)
    topk_labels = []
    smoothed_scores = scores if bw is None else gaussian_filter(scores, bw)
    for th in thresholds:
        _rst = [(idx, binary, s) for idx, (binary, s) in enumerate(zip(smoothed_scores > th, scores))]
        topk_labels.extend(_rst)

    # print("topk_labels {}, {}".format(len(topk_labels), topk_labels[0]))

    bboxes = []
    tol_lst = [0.05, .1, .2, .3, .4, .5, .6, 0.8, 1.0]
    # tol_lst = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.12, 0.15, .2, .3, .4, .5, .6, 0.8, 1.0]
    bboxes.extend(build_box_by_search_with_id(topk_labels, np.array(tol_lst)))

    bboxes = np.array(bboxes)

    # print("bboxes before nms {}".format(len(bboxes)))
    bboxes = find_unique_rows_with_lexsort(bboxes)
    # if len(bboxes) > 1:
    #     bboxes = temporal_nms(bboxes, 0.99)
    # print("bboxes after nms {}".format(len(bboxes)))

    # filter out too short or too long proposals
    bboxes = bboxes[(bboxes[:, 2] - bboxes[:, 1] >= min_len) * (bboxes[:, 2] - bboxes[:, 1] <= max_len)]
    # bboxes = list(filter(lambda b: max_len >= b[1] - b[0] >= min_len, bboxes))
    # print("bboxes after filter {}".format(len(bboxes)))

    # each tuple is [st_frm_idx, ed_frm_idx, score], sort by scores
    # pr_box = sorted(bboxes, key=lambda b: b[2], reverse=True)  # decreasing order
    # pr_box_list = [bboxes[bboxes[:, 0] == idx][:, 1:] ddddddfor idx in range(len(scores))]
    # pr_box_list = [bboxes[e[:, 2].argsort()[::-1]] for e in pr_box_list]
    _bboxes = []
    for i, (video_idx, video_score) in enumerate(zip(video_indices, video_scores)):
        selected = bboxes[bboxes[:, 0] == i]
        selected[:, 0] = video_idx
        selected[:, 3] *= video_score
        _bboxes.append(selected)
    bboxes = np.concatenate(_bboxes, axis=0)
    pr_box = bboxes[bboxes[:, 3].argsort()[::-1]][:max_out]  # decreasing order
    pr_box[:, 2] += 1
    pr_box[:, 1:3] = pr_box[:, 1:3] * clip_length
    # print("pr_box", pr_box[:10])
    # raise NotImplementedError
    pr_box = [e.tolist() for e in pr_box]
    return pr_box


def scale_to_range01(scores):
    """scale to range [0, 1]
    scores: np.ndarray of shape (N, *).
        (scores - min(scores))
    """
    scores = scores - np.min(scores, )
    scores = scores / np.max(scores)
    return scores



