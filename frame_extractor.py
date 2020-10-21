import os
import av
from glob import glob
import ray
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--videos-path", type=str, default="../Data/UCF101/UCF101_videos/")
parser.add_argument("--frames-path", type=str, default="../Data/UCF101/UCF101_frames/")
parser.add_argument("--num-cpus", type=int, default=8)
args = parser.parse_args()

# check directory
assert os.path.exists(args.frames_path) == False, "'{}' directory is alreay exist !!".format(args.frames_path)
os.makedirs(args.frames_path)

# init ray on local
ray.init(num_cpus=args.num_cpus)

videos_path_list = glob(os.path.join(args.videos_path, "*.avi"))
print("will be start extract frames from {} videos...".format(len(videos_path_list)))

@ray.remote
def extractor(index, video_path):
    frame_name = video_path.split("\\" if os.name == 'nt' else "/")[-1].split('.avi')[0]
    frame_path = os.path.join(args.frames_path, frame_name)
    frames = [f.to_image() for f in av.open(video_path).decode(0)]
    os.makedirs(frame_path)
    for j, frame in enumerate(frames):
        frame.save(os.path.join(frame_path, "{}.jpg".format(j)))
    print("{}/{} extracted {} frames from '{}'".format(index, len(videos_path_list), len(frames), frame_name + '.avi'))

features = [extractor.remote(i, video_path) for i, video_path in enumerate(videos_path_list)]
ray.get(features)
