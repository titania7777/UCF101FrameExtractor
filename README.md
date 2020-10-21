# UCF101 Frame Extractor with ray

## Requirements

*   ray>=0.8.7
*   av>=8.0.2

## Usage

download raw videos [Download UCF101.tar](https://www.dropbox.com/s/xhwkilwgytox0j2/UCF101.tar?dl=0) and extract frames from videos
```
frame_extractor.py --videos-path /path/to/videos --frames-path /path/to/save --num-cpus /number/of/cpus
```