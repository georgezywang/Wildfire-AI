import argparse
from WildfireAI.Model import TrainModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputdir", help = "input directory")
    parser.add_argument("-o", "--outputdir", help = "directory to save model and checkpoint")
    parser.add_argument("-m", "--mask", help = "use segmentation", required = False, action = "store_true")
    parser.add_argument("-v", "--visualize", help = "visualize data", required = False, action = "store_true")
    args = parser.parse_args()
    inputDir = args.inputdir
    outputDir = args.outputdir
    mask = args.mask
    visualize = args.visualize
    TrainModel(inputDir, outputDir, mask, visualize)
