import os
import ray
import numpy as np
from glob import glob
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--videos-path", type=str, default="../Data/UCF101/UCF101_videos/")
parser.add_argument("--frames-path", type=str, default="./UCF101_frames/")
parser.add_argument("--flows-path", type=str, default="./UCF101_flows/")
parser.add_argument("--flow-mode", action="store_true")
parser.add_argument("--num-cpus", type=int, default=8)
parser.add_argument("--pyr-scale", type=float, default=0.5)
parser.add_argument("--levels", type=int, default=3)
parser.add_argument("--winsize", type=int, default=15)
parser.add_argument("--iterations", type=int, default=3)
parser.add_argument("--poly-n", type=int, default=5)
parser.add_argument("--poly-sigma", type=float, default=1.2)
parser.add_argument("--flags", type=int, default=0)
args = parser.parse_args()

# check directory(videos_path, frames_path, flows_path)
assert os.path.exists(args.videos_path) is True, "'{}' directory is not exist !!".format(args.videos_path)

# only flow
if args.flow_mode:
    assert os.path.exists(args.flows_path) is False, "'{}' directory is already exist !!".format(args.flows_path)
else:
    assert os.path.exists(args.frames_path) is False, "'{}' directory is already exist !!".format(args.frames_path)

# get videos path
videos_path_list = glob(os.path.join(args.videos_path, "*"))

# init ray on local
ray.init(num_cpus=args.num_cpus)

@ray.remote
def dense_optical_flow(index, video_path):
    frame_name = video_path.split("\\" if os.name == 'nt' else "/")[-1].split('.avi')[0]
    
    # read video
    cap = cv2.VideoCapture(video_path)
    ret, frame_first = cap.read()
    if ret == False:
        print("'{}' video read to fail !! skip this video...".format(video_path))
        return

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("{}/{} name: {} length: {}".format(index+1, len(videos_path_list), frame_name, length))

    # only flow
    if args.flow_mode:
        # flow path
        flow_path = os.path.join(args.flows_path, frame_name)
        os.makedirs(flow_path)

        # convert to gray
        frame_prev_gray = cv2.cvtColor(frame_first, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame_first)
        hsv[..., 1] = 255
    else:
        # frame path
        frame_path = os.path.join(args.frames_path, frame_name)
        os.makedirs(frame_path)

        # save first frame
        if not cv2.imwrite(os.path.join(frame_path, "0.jpg"), frame_first):
            raise Exception("could not write frame !!")

    for i in range(1, length):
        # read next frame
        ret, frame_next = cap.read()
        if ret == False:
            msg = "index '{}' of '{}' video read to fail !! skip this frame...".format(i, video_path)
            continue
        
        # only flow
        if args.flow_mode:
            frame_next_gray = cv2.cvtColor(frame_next, cv2.COLOR_BGR2GRAY)
            # Computes a dense optical flow using the Gunnar Farneback's algorithm
            frame_flow = cv2.calcOpticalFlowFarneback(frame_prev_gray, frame_next_gray, None, args.pyr_scale, args.levels, args.winsize, args.iterations, args.poly_n, args.poly_sigma, args.flags)
            # Calculates the magnitude and angle of 2D vectors
            mag, ang = cv2.cartToPolar(frame_flow[..., 0], frame_flow[..., 1])
            hsv[..., 0] = ang*180/np.pi/2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite(os.path.join(flow_path, "{}.jpg".format(i - 1)), cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
            frame_prev_gray = frame_next_gray
        else:
            # save next frame 
            if not cv2.imwrite(os.path.join(frame_path, "{}.jpg".format(i)), frame_next):
                raise Exception("could not write frame !!")
    cap.release()

ray.get([dense_optical_flow.remote(i, video_path) for i, video_path in enumerate(videos_path_list)])