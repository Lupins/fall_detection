import argparse
import os
import json
import subprocess

def save_clips_from_json(json_path, threshold):
    with open(json_path) as json_file:
        data = json.load(json_file)
    video_path = data['video']['path']
    filename = os.path.basename(video_path)
    folder = os.path.dirname(os.path.abspath(video_path))
    save_path = os.path.join(folder, filename.split('.')[0] + '_clips')
    if os.path.exists(save_path) == False:
        os.mkdir(save_path)
    for inference in data['inferences']:
        if float(inference['probability']) > float(threshold):
            clip_name = os.path.join(save_path, inference['start_time'] + '_' + inference['end_time'] + '.mkv')
            print(clip_name)
            subprocess.call(['ffmpeg', '-ss', inference['start_time'], '-to',
                             inference['end_time'], '-i', video_path, '-map_metadata',
                             '-1', '-c', 'copy', clip_name])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help = "JSON file with model inferences.", required=True)
    parser.add_argument("-t", "--threshold", help = "Threshold limit for saving clips.", default=0.5)
    args = parser.parse_args()
    json_path = args.input
    threshold = args.threshold

    save_clips_from_json(json_path, threshold)