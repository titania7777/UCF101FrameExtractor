# UCF101 Frame Extractor with ray
this frame extractor supports optical flow(dense optical flow) also

## Requirements

*   ray>=0.8.7
*   opencv-python>=4.4.0.44

## Usage

download raw videos [Download UCF101.tar](https://www.dropbox.com/s/xhwkilwgytox0j2/UCF101.tar?dl=0) and decompress it

extract only frames
```
frame_extractor.py --videos-path /path/to/videos --frames-path /path/to/save --num-cpus /number/of/cpus
```

extract both (frames, flows)
```
frame_extractor.py --videos-path /path/to/videos --frames-path /path/to/save --flows-path /path/to/save --num-cpus /number/of/cpus --flow-mode
```