import MaskCoco
import os
import json
import argparse

def fileDic(path):
    imgList = []
    for i in range(501):
        mask = str(i).zfill(3) + "_gt.png"
        img = str(i).zfill(3) + "_rgb.png"
        maskPath = os.path.join(path, mask)
        if os.path.exists(maskPath):
            img = MaskCoco.ImageLabel(img, mask, 0, i)
            imgList.append(img)
    return imgList

def readDict(path):
    with open(path, 'r') as file:
        catDic = json.load(file)
    return catDic

def checkArgs(args):
    assert os.path.exists(args.inpath), "invalid input path"
    assert os.path.exists(args.catpath), "invalid category dictionary path"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    fileDir = os.path.dirname(__file__)
    parser.add_argument("-i", "--inpath", help = "directory of input images")
    parser.add_argument("-o", "--outpath", help = "coco json output file path", required = False, default = os.path.join(fileDir, "data.json"))
    parser.add_argument("-c", "--catpath", help = "path of category dictionary json file")
    parser.add_argument("-d", "--imageid", help = "specify if use image id", action='store_true', required = False)
    args = parser.parse_args()
    checkArgs(args)

    catDic = readDict(args.catpath)
    imgList = fileDic(args.inpath)
    myParser = MaskCoco.MaskParser(catDic, imgList, inPath = args.inpath, outPath=args.outpath, useImgID = args.imageid)
    myParser.saveJson()
