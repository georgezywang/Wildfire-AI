import torch, torchvision
import numpy as np
import os, json, cv2, random

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import Visualizer

class CocoTrainer(DefaultTrainer):
  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
    return COCOEvaluator(dataset_name, cfg, False, output_folder)

def visualizeImage(dataset):
    my_dataset_train_metadata = MetadataCatalog.get(dataset)
    dataset_dicts = DatasetCatalog.get(dataset)
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow(vis.get_image()[:, :, ::-1])

def TrainModel(inputDir, outputDir, mask = True, visualize = False):
    trainDat = os.path.join(inputDir, "train")
    valDat = os.path.join(inputDir, "val", "val.json")
    testDat = os.path.join(inputDir, "test", "test.json")
    register_coco_instances("my_dataset_train", {}, os.path.join(trainDat, "train.json"), trainDat)
    register_coco_instances("my_dataset_val", {}, os.path.join(valDat, "val.json"), valDat)
    register_coco_instances("my_dataset_test", {}, os.path.join(testDat, "test.json"), testDat)

    if (visualize):
        visualizeImage("my_dataset_train")

    cfg = get_cfg()
    if (mask):
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    else:
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 4 
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.MAX_ITER = 2000 
    cfg.SOLVER.STEPS = (1000, 1500)
    cfg.SOLVER.GAMMA = 0.05
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.TEST.EVAL_PERIOD = 500
    cfg.OUTPUT_DIR = outputDir

    os.makedirs(cfg.OUTPUT_DIR, exist_ok = True)
    trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume = True)
    trainer.train()

    if (visualize):
        visualizeImage("my_dataset_test")

    evaluator = COCOEvaluator("my_dataset_test", ("bbox", "segm"), False)
    val_loader = build_detection_test_loader(cfg, "my_dataset_test")
    print(inference_on_dataset(trainer.model, val_loader, evaluator))

    with open(os.path.join(outputDir, "config.yaml"), 'w') as f:
        f.write(cfg.dump())
