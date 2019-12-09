"""Script for I3D model inference"""

from soccer_functions import game_inference
from keras.models import load_model
import argparse
from natsort import natsorted
import keras_metrics
import glob2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main(input_path, model_path):
    
    model = load_model(model_path, custom_objects={"binary_precision":keras_metrics.binary_precision(), "binary_recall":keras_metrics.binary_recall()})
    
    allowed_filetypes = ['.avi', '.wmv', '.mpg', '.mov', '.mp4', '.mkv', '.3gp', '.webm', '.ogv']
    videos_path = []

    if os.path.isfile(input_path) and input_path.lower().endswith(tuple(allowed_filetypes)):
        videos_path = [input_path]
        
    elif os.path.isdir(input_path):
        search_path = os.path.join(input_path, '**')
        for ext in allowed_filetypes:
            videos_path.extend(glob2.glob(os.path.join(search_path, '*' + ext)))
    else:
        print("Not a valid input.")

    videos_path = natsorted(videos_path)
    for video_path in videos_path:
        folder = os.path.dirname(os.path.abspath(video_path))
        filename = os.path.basename(video_path)
        json_path = os.path.join(folder, filename.split('.')[0] + '.json')
        print(video_path)
        if os.path.isfile(json_path):
            continue
        else:
            game_inference(model, video_path, json_path, samples = 24, height = 224, width = 224)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help = "Video path or folder for model inference.", required=True)
    parser.add_argument("-m", "--model", help = "Model path.", required=True)
    args = parser.parse_args()
    input_path = args.input
    model_path = args.model

    main(input_path, model_path)