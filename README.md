# UCF101 Frame Extractor with ray
this frame extractor supports optical flow(dense optical flow) also

## Requirements

*   ray>=0.8.7
*   opencv-python>=4.4.0.44

## Usage
example
```
$ wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
$ mkdir UCF101_videos
$ unrar e UFC101.rar ./UCF101_videos
$ python frame_extractor.py --videos-path ./UCF101_videos --frames-path ./UCF101_frames --num-cpus 8
```

extract only frames
```
frame_extractor.py --videos-path /path/to/videos --frames-path /path/to/save --num-cpus /number/of/cpus
```

extract only optical flows
```
frame_extractor.py --videos-path /path/to/videos --flows-path /path/to/save --num-cpus /number/of/cpus --flow-mode
```