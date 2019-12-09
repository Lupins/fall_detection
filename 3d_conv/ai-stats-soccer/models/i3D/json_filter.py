"""Script for joining inferences that are temporal close each other"""

from soccer_functions import updateJsonFile
import argparse
import os
import json
import time
import datetime
import numpy as np

def find_duplicates(itens, window=1):
    idxs = np.arange(len(itens))
    for i, diff in enumerate(np.ediff1d(itens)):
        if diff > window:
            idxs[i+1] = idxs[i] + 1
        else:
            idxs[i+1] = idxs[i]
    return idxs

def remove_duplicates_json(json_path, window):
    with open(json_path) as json_file:
        data = json.load(json_file)

    dirname = os.path.dirname(json_path)
    new_name = os.path.basename(json_path).split('.')[0] + '_filtered.json'
    new_json_path = os.path.join(dirname, new_name)

    new_data = {}
    new_data['video'] = data['video']
    new_data['inferences'] = []

    with open(new_json_path, 'w') as outfile:
        json.dump(new_data, outfile)

    actions = []
    probabilities = []
    start_times = []
    end_times = []
    start_frames = []
    end_frames = []
    for inference in data['inferences']:
        start_time = time.strptime(inference['start_time'], '%H:%M:%S')
        end_time = time.strptime(inference['end_time'], '%H:%M:%S')
        start_seconds = int(datetime.timedelta(hours=start_time.tm_hour, minutes=start_time.tm_min, seconds=start_time.tm_sec).total_seconds())
        end_seconds = int(datetime.timedelta(hours=end_time.tm_hour, minutes=end_time.tm_min, seconds=end_time.tm_sec).total_seconds())
        actions.append(inference['action'])
        probabilities.append(inference['probability'])
        start_frames.append(inference['start_frame'])
        end_frames.append(inference['end_frame'])
        start_times.append(start_seconds)
        end_times.append(end_seconds)

    duplicated_idx = find_duplicates(start_times, window)

    for i, idx in enumerate(np.unique(duplicated_idx)):
        start = min(np.array(start_times)[duplicated_idx == idx])
        end = max(np.array(end_times)[duplicated_idx == idx])
        start_frame = min(np.array(start_frames)[duplicated_idx == idx])
        end_frame = max(np.array(end_frames)[duplicated_idx == idx])
        result = max(np.array(probabilities, dtype=np.float)[duplicated_idx == idx])
        updateJsonFile(new_json_path, result, start, end, start_frame, end_frame)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help = "JSON file with model inferences.", required=True)
    parser.add_argument("-w", "--window", help = "Minimum window for joining clips.", default=1)
    args = parser.parse_args()
    json_path = args.input
    window = args.window

    remove_duplicates_json(json_path, window)