REPO = "detectron2"
if [ ! -d $REPO ]
then
    git clone https://github.com/facebookresearch/detectron2
fi

python3 driver.py -i FireData -o FireData/Output -m
python3 detectron2/demo/demo.py --config-file FireData/Output/Config.yaml  --video-input FireData/video.mp4 --confidence-threshold 0.7 --output FireData/video-mask.mkv --opts MODEL.WEIGHTS FireData/Output/model_final.pth