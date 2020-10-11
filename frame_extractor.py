import os
import av
import glob
import ray
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--videos-path", type=str, default="./UCF101_videos/")
parser.add_argument("--frames-path", type=str, default="./UCF101_frames/")
args = parser.parse_args()

ray.init(num_cpus=8)

videos_path_list = glob.glob(args.videos_path + '*.avi')
print("will start extract frames from {} videos...".format(len(videos_path_list)))

# check directory
assert os.path.exists(args.frames_path) is False, "'{}' directory is alreay exist!!".format(args.frames_path)
os.makedirs(args.frames_path)

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