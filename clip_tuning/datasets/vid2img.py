import torch as th
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import ffmpeg
import math
import csv
import time
import argparse
import json
import tqdm
import joblib

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

def vid2anujvid(vid):
    temp = vid.split("_")
    start, end = temp[-2], temp[-1]
    suffix = "_" + start + "_" + end
    ytvid = vid[:-len(suffix)]
    start, end = float(start), float(end)
    anuj_vid = ytvid + "_" + str(int(start)) + "_" + str(int(end))
    return anuj_vid

def _get_output_dim(size, h, w):
    if isinstance(size, tuple) and len(size) == 2:
        return size
    elif h >= w:
        return int(h * size / w), size
    else:
        return size, int(w * size / h)

def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        try:
            num, denom = frac_str.split('/')
        except ValueError:
            return None
        try:
            leading, num = num.split(' ')
        except ValueError:
            return float(num) / float(denom)
        if float(leading) < 0:
            sign_mult = -1
        else:
            sign_mult = 1
        return float(leading) + sign_mult * (float(num) / float(denom))
def _get_video_info(video_path):
    probe = ffmpeg.probe(video_path)
    video_stream = next((stream for stream in probe['streams']
                            if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    fps = math.floor(convert_to_float(video_stream['avg_frame_rate']))
    try:
        frames_length = int(video_stream['nb_frames'])
        duration = float(video_stream['duration'])
    except Exception:
        frames_length, duration = -1, -1
    info = {"duration": duration, "frames_length": frames_length,
            "fps": fps, "height": height, "width": width}
    return info

def vid2img(item, args, save_root):
    vid = item['vid']
    anuj_vid = vid2anujvid(vid)

    video_path1 = os.path.join(args.video_root, anuj_vid+".mkv")
    if os.path.isfile(video_path1):
        video_path = video_path1
    else:
        video_path = os.path.join(args.video_root, vid+".mp4")
        assert os.path.isfile(video_path)
    out_root = os.path.join(save_root, vid)
    if not args.overwrite:
        load_flag = (not os.path.isdir(out_root)) or (not os.path.isfile(out_root+"/0.jpg"))
    if load_flag:
        os.makedirs(out_root, exist_ok=True)
        try:
            info = _get_video_info(video_path)
            h, w = info["height"], info["width"]
            # print(f"original video size: {h}x{w}")
        except Exception:
            print('ffprobe failed at: {}'.format(video_path))
            # return {'video': th.zeros(1), 'input': video_path,
            #         'output': output_file, 'info': {}}
        height, width = _get_output_dim(args.size, h, w)
        # print(f"scale the video to {height}x{width}")
        try:
            duration = info["duration"]
            fps = args.framerate
            if duration > 0 and duration < 1/fps+0.1:
                fps = 2/max(int(duration), 1)
                print(duration, fps)
        except Exception:
            fps = args.framerate
        all_video = []
        all_len = []
        cmd = (
                ffmpeg
                .input(video_path)
                .filter('fps', fps=fps)
                .filter('scale', width, height)
                )
        if args.centercrop:
            x = int((width - args.size) / 2.0)
            y = int((height - args.size) / 2.0)
            cmd = cmd.crop(x, y, args.size, args.size)
            # print(f"x: {x}, y:{y}, size: {args.size}")
        out, _ = (
            cmd.output(out_root+'/'+'%d.jpg', start_number=0)
            .run(quiet=True)
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Easy video feature extractor')
    parser.add_argument("--jsonl_root", type=str, default="/home/pyp/vqhighlight/data")
    parser.add_argument("--split", type=str, choices=['val', 'test', 'train'])
    parser.add_argument("--video_root", type=str, default="/saltpool0/data/pyp/vqhighlight/video")
    parser.add_argument("--save_root", type=str, default="/saltpool0/data/pyp/vqhighlight/image")
    parser.add_argument("--framerate", type=int, default=1)
    
    parser.add_argument("--size", type=int, default=256, help="short side length of every frame")
    parser.add_argument("--centercrop", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--parallelize", action="store_true", default=False)
    
    args = parser.parse_args()
    
    save_root = os.path.join(args.save_root, "framerate"+str(args.framerate))
    os.makedirs(save_root, exist_ok=True)
    data = load_jsonl(os.path.join(args.jsonl_root, 'highlight_'+args.split+"_release.jsonl"))
    if args.parallelize:
        parallizer = joblib.Parallel(n_jobs=64, max_nbytes=None, verbose=2)
        watershed_windows_out = parallizer(joblib.delayed(vid2img)(item, args, save_root) for item in data)
    else:
        for item in tqdm.tqdm(data):
            vid = item['vid']
            anuj_vid = vid2anujvid(vid)

            video_path1 = os.path.join(args.video_root, anuj_vid+".mkv")
            if os.path.isfile(video_path1):
                video_path = video_path1
            else:
                video_path = os.path.join(args.video_root, vid+".mp4")
                assert os.path.isfile(video_path)
            out_root = os.path.join(save_root, vid)
            if not args.overwrite:
                load_flag = (not os.path.isdir(out_root)) or (not os.path.isfile(out_root+"/0.jpg"))

            if load_flag:
                os.makedirs(out_root, exist_ok=True)
                try:
                    info = _get_video_info(video_path)
                    h, w = info["height"], info["width"]
                    # print(f"original video size: {h}x{w}")
                except Exception:
                    print('ffprobe failed at: {}'.format(video_path))
                    # return {'video': th.zeros(1), 'input': video_path,
                    #         'output': output_file, 'info': {}}
                height, width = _get_output_dim(args.size, h, w)
                # print(f"scale the video to {height}x{width}")
                try:
                    duration = info["duration"]
                    fps = args.framerate
                    if duration > 0 and duration < 1/fps+0.1:
                        fps = 2/max(int(duration), 1)
                        print(duration, fps)
                except Exception:
                    fps = args.framerate
                all_video = []
                all_len = []
                cmd = (
                        ffmpeg
                        .input(video_path)
                        .filter('fps', fps=fps)
                        .filter('scale', width, height)
                        )
                if args.centercrop:
                    x = int((width - args.size) / 2.0)
                    y = int((height - args.size) / 2.0)
                    cmd = cmd.crop(x, y, args.size, args.size)
                    # print(f"x: {x}, y:{y}, size: {args.size}")
                out, _ = (
                    cmd.output(out_root+'/'+'%d.jpg', start_number=0)
                    .run(quiet=True)
                )
                # if args.centercrop and isinstance(args.size, int):
                #     height, width = args.size, args.size
                # video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
                # video = video.transpose((0, 3, 1, 2))
                # print(f"final video frame size: {video.shape[-2]}x{video.shape[-1]}")
                # np.save(out_fn, video)
                # assert False