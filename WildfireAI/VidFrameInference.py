import imageio, cv2, os, json
import numpy as np
import glob

from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer

def GetFrames(video, outputDir):
    reader = imageio.get_reader(video)
    print("Extracting Frames...")
    if not os.path.exists(os.path.join(outputDir, "vidFrames")):
        os.makedirs(os.path.join(outputDir, "vidFrames"))
    for frame_number, im in enumerate(reader):
        frame_number = str(frame_number).zfill(5)
        imageio.imwrite(os.path.join(outputDir, "vidFrames", "frame_{}.jpg".format(frame_number)), im)

def RunInference(modelDir, outputDir, visualize = False):
    with open(os.path.join(outputDir, "vidFrames", "dat.json"), 'w') as f:
        json.dump("", f)
    register_coco_instances("vidFrames", {}, os.path.join(outputDir, "vidFrames", "dat.json"), os.path.join(outputDir, "vidFrames"))
    cfg = get_cfg()
    cfg.load_cfg(os.path.join(modelDir, "config.yaml"))
    cfg.DATASETS.TEST = ("vidFrames", )
    predictor = DefaultPredictor(cfg)
    test_metadata = MetadataCatalog.get("vidFrames")

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
    count = 0
    for imageName in glob.glob('{}/.*jpg'.format(outputDir)):
        im = cv2.imread(imageName)
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                        metadata=test_metadata, 
                        scale=0.8
                        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        if (visualize):
            cv2.imshow(out.get_image()[:, :, ::-1])
        frame_num = str(count).zfill(5)
        cv2.imwrite(os.path.join(outputDir, "vidFrames", "frame_{}.jpg".format(frame_number)), out.get_image()[:, :, ::-1])
        count = count + 1
        print("Inferencing frame {}".format(frame_num))
 
def ConstructVideo(inputDir, videoDir):
    print("Composing video...")
    img_array = []
    for filename in glob.glob('{}/.*jpg'.format(inputDir)):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def InferenceVideo(video, modelDir, frameDir, segmentDir, outputVid, visualize = False):
    GetFrames(video, frameDir)
    RunInference(modelDir, segmentDir, visualize)
    ConstructVideo(segmentDir, outputVid)
