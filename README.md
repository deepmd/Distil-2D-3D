# 2D to 3D Knowledge Distillation

## Summary

This is the PyTorch code for 2d to 3d distillation experiments.

## Requirements

* [PyTorch](http://pytorch.org/)

```bash
pip install pytorch torchvision
```

* FFmpeg, FFprobe

```bash
wget http://johnvansickle.com/ffmpeg/releases/ffmpeg-release-64bit-static.tar.xz
tar xvf ffmpeg-release-64bit-static.tar.xz
cd ./ffmpeg-3.3.3-64bit-static/; sudo cp ffmpeg ffprobe /usr/local/bin;
```

* Other

```bash
pip install PyYAML Pillow scikit-learn scikit-video pandas tqdm progress tensorboardX
```
**Note**: scikit-video has some issues with ffmpeg 3.4.x but works flawlessly with ffmpeg 2.8.15. 
Also note that installing opencv 3.3 and upper in the a conda environment, will install ffmpeg 3.4 as well.  

## Preparation

### ActivityNet

* Download videos using [the official crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler).
* Convert from avi to jpg files using ```tools/video_jpg.py```

```bash
python tools/video_jpg.py avi_video_directory jpg_video_directory
```

* Generate fps files using ```tools/fps.py```

```bash
python tools/fps.py avi_video_directory jpg_video_directory
```

### Kinetics

* Download videos using [the official crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics).
  * Locate test set in ```video_directory/test```.
* Convert from avi to jpg files using ```tools/video_jpg_kinetics.py```

```bash
python tools/video_jpg_kinetics.py avi_video_directory jpg_video_directory
```

* Generate n_frames files using ```tools/n_frames_kinetics.py```

```bash
python tools/n_frames_kinetics.py jpg_video_directory
```

* Generate annotation file in json format similar to ActivityNet using ```tools/kinetics_json.py```
  * The CSV files (kinetics_{train, val, test}.csv) are included in the crawler.

```bash
python tools/kinetics_json.py train_csv_path val_csv_path test_csv_path dst_json_path
```

### UCF-101

* Download videos and train/test splits [here](http://crcv.ucf.edu/data/UCF101.php).
* Convert from avi to jpg files using ```tools/video_jpg_ucf101_hmdb51.py```

```bash
python tools/video_jpg_ucf101_hmdb51.py avi_video_directory jpg_video_directory
```

* Generate n_frames files using ```tools/n_frames_ucf101_hmdb51.py```

```bash
python tools/n_frames_ucf101_hmdb51.py jpg_video_directory
```

* Generate annotation file in json format similar to ActivityNet using ```tools/ucf101_json.py```
  * ```annotation_dir_path``` includes classInd.txt, trainlist0{1, 2, 3}.txt, testlist0{1, 2, 3}.txt

```bash
python tools/ucf101_json.py annotation_dir_path
```

### HMDB-51

* Download videos and train/test splits [here](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/).
* Convert from avi to jpg files using ```tools/video_jpg_ucf101_hmdb51.py```

```bash
python tools/video_jpg_ucf101_hmdb51.py avi_video_directory jpg_video_directory
```

* Generate n_frames files using ```tools/n_frames_ucf101_hmdb51.py```

```bash
python tools/n_frames_ucf101_hmdb51.py jpg_video_directory
```

* Generate annotation file in json format similar to ActivityNet using ```tools/hmdb51_json.py```
  * ```annotation_dir_path``` includes brush_hair_test_split1.txt, ...

```bash
python tools/hmdb51_json.py annotation_dir_path
```

## Running the code

Run knowledge-distillation train/eval:

```bash
python main.py --config yaml_config_path
```


