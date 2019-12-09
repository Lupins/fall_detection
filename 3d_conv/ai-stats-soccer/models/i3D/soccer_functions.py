import os
import cv2
import glob
import numpy as np
import pandas as pd
from natsort import natsorted
import argparse
import keras.backend as K
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib.animation
from inspect import signature
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tqdm import tqdm
import datetime
import json
import math

pd.options.mode.chained_assignment = None

def load_metrics(threshold=0.5):
    total_acc = metric_generator('accuracy', threshold=threshold)
    total_acc.__name__ = 'accuracy'

    total_precision = metric_generator('precision', threshold=threshold)
    total_precision.__name__ = 'precision'

    total_recall = metric_generator('recall', threshold=threshold)
    total_recall.__name__ = 'recall'

    total_f1 = metric_generator('f1', threshold=threshold)
    total_f1.__name__ = 'f1'
    
    return total_acc, total_precision, total_recall, total_f1

def eval_thresholds(y_true, y_pred):
    for threshold in np.linspace(0,1,41):
        accuracy = accuracy_score(y_true, y_pred >= threshold)
        precision = precision_score(y_true, y_pred >= threshold)
        recall = recall_score(y_true, y_pred>=threshold)
        f1 = (precision + recall)/2
        print("Threshold = %.2f, Accuracy = %.2f, Precision = %.3f, Recall = %.3f, F1 = %.3f"%(threshold, accuracy, precision, recall, f1))


def precision_recall_curve(y_true, y_pred):
    average_precision = average_precision_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    f1_scores = []
    for thresh in thresholds:
        f1_scores.append(f1_score(y_true, [1 if m > thresh else 0 for m in y_pred]))

    f1_scores = np.array(f1_scores)
    max_f1 = f1_scores.max() 
    max_f1_threshold =  thresholds[f1_scores.argmax()]
    
    step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
    plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    plt.grid()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP=%0.3f, MaxF1=%0.3f, MaxF1_thresh = %0.3f'%(average_precision, max_f1, max_f1_threshold))
    plt.show();
    
    return max_f1_threshold

def split_video(video_id, num_clips=0, seed=10, save_path = 'splits/', videos_path = 'videos/'):
    main_directory = os.path.abspath(save_path + 'split_' + str(video_id))
    if not os.path.exists(main_directory):
        os.makedirs(main_directory)
            
    cap = cv2.VideoCapture(glob.glob(videos_path + '*/*' + str(video_id) + '*.mp4')[0])
    fps = cap.get(cv2.CAP_PROP_FPS)
    clip_length = int(fps*4)
    samples = 24
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = 224
    width = 224
    video = np.zeros(shape=(video_length, height, width, 3), dtype=np.uint8)

    if num_clips > 0:
        np.random.seed(seed)
        splits_start = np.random.randint(video_length-clip_length, size=num_clips)
        for split_start in splits_start:
            clip_name = str(video_id) + '_' + str(split_start) + '_' + str(split_start + clip_length)
            directory = os.path.join(main_directory, clip_name)
            selected_idx = np.linspace(split_start, split_start + clip_length, samples).astype(int)
            selected_frames = np.empty(shape=(len(selected_idx), height, width, 3), dtype=np.uint8)
            for i, idx in enumerate(selected_idx):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                frame = cv2.resize(cap.read()[1], (width,height))
                selected_frames[i] = frame
            directory = os.path.join(main_directory, clip_name)
            np.save(directory, selected_frames)
        
    else:
        print('Starting video resizing')
        for i in range(video_length):
            video[i] = cv2.resize(cap.read()[1], (width,height))
        splits_start = np.arange(0, video_length, 25)
        print('Starting saving splits')
        for i in splits_start:
            if i + clip_length < video_length:
                clip_name = str(video_id) + '_' + str(i) + '_' + str(i + clip_length)
                directory = os.path.join(main_directory, clip_name)
                selected_idx = np.linspace(i, i + clip_length, samples).astype(int)
                selected_frames = video[selected_idx]
                np.save(directory, selected_frames)

def dataframe_from_splits(video_id, dataframe, split_path = 'splits/', threshold = 0.5):
    video_id = str(video_id)
    video_df = dataframe[dataframe['video'] == video_id]
    clips = glob.glob(split_path + '/split_'+ video_id + '/*.npy')
    videos = []
    starts = []
    ends = []
    for filename in clips:
        videos.append(filename.split('/')[-1].split('_')[0])
        starts.append(filename.split('/')[-1].split('_')[1])
        ends.append(filename.split('/')[-1].split('_')[-1].split('.')[0])

    clips_dataframe = pd.DataFrame(np.c_[videos, starts, ends], columns= ['video', 'start', 'end']) 
    clips_dataframe["start"] = pd.to_numeric(clips_dataframe["start"])
    clips_dataframe["end"] = pd.to_numeric(clips_dataframe["end"])
    clips_dataframe = clips_dataframe.sort_values(by=['start'])
    
    clips_actions = pd.DataFrame()
    for clip_idx in range(len(clips_dataframe)):
        clip_start = int(clips_dataframe.iloc[clip_idx]['start'])
        clip_end = int(clips_dataframe.iloc[clip_idx]['end'])
        intersection_dataframe = video_df[~((clip_end < video_df['start'].astype(int)) | (clip_start > video_df['end'].astype(int)))]
        action_duration = intersection_dataframe['end'] - intersection_dataframe['start']
        if len(intersection_dataframe) > 0:
            lastest_start = np.maximum(intersection_dataframe['start'], clip_start)
            firstest_end = np.minimum(intersection_dataframe['end'], clip_end)
            overlap = firstest_end - lastest_start
            proportion = overlap/action_duration
            valid = (overlap/action_duration) >= threshold
            if np.sum(valid) > 0:
                slice_df = clips_dataframe.iloc[clip_idx:clip_idx+1]
                slice_df['type'] = [intersection_dataframe.loc[valid.index[0]]['type']]
                clips_actions = clips_actions.append(slice_df, ignore_index=True)
                continue
        slice_df = clips_dataframe.iloc[clip_idx:clip_idx+1]
        slice_df['type'] = 'none'
        clips_actions = clips_actions.append(slice_df, ignore_index=True)
        
    return clips_actions


def filter_clips(dataframe, path = 'train_clips/'):
    files = glob.glob(path + '/*.npy')
    id_list = []
    for file in files:
        video_id, start_frame, end_frame = file.split('/')[-1].split('.npy')[0].split('_')
        aux_df = dataframe[(dataframe['video'] == video_id) & (dataframe['start'] == int(start_frame)) & (dataframe['end'] == int(end_frame))]
        if len(aux_df) > 0:
            ID = aux_df.iloc[0].name
            id_list.append(ID)

    id_list.extend(list(dataframe[dataframe['type'] == 'none'].index))
    dataframe = dataframe.loc[id_list]
    return dataframe


def confusion_matrix(y_pred, y_true, columns, norm=True):
    matrix = metrics.multilabel_confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    if norm == True:
        matrix = matrix/(np.expand_dims(matrix.sum(axis=1), -1) + 10e-6)
    df_cm = pd.DataFrame(matrix, index = columns, columns = columns)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, cmap=sn.cm.rocket_r)
    ap = metrics.average_precision_score(y_true, y_pred)
    acc = metrics.accuracy_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    print("Accuracy: %s"%acc)
    print("Average Precision: %s"%ap)

def metric_generator(metric, threshold=0.5):
    
    def metrics(y_true, y_pred):
        y_true_flatten = K.cast(K.flatten(y_true),'bool')
        y_pred_flatten = K.cast(K.flatten(y_pred),'float32')
        y_true_mask = K.cast(y_true_flatten,'float32')
        y_true_bg = K.cast(~y_true_flatten,'float32')
        y_pred_mask = K.cast(y_pred_flatten >= threshold,'float32')
        y_pred_bg = K.cast(~(y_pred_flatten >= threshold),'float32')

        tp = K.sum(y_true_mask*y_pred_mask)
        fp = K.sum(y_true_bg*y_pred_mask)
        tn = K.sum(y_true_bg*y_pred_bg)
        fn = K.sum(y_true_mask*y_pred_bg)

        if metric == 'accuracy':
            accuracy = (tp+tn)/K.clip(tp+fp+tn+fn, K.epsilon(), None)
            return accuracy
        elif metric == 'precision':
            precision = tp/K.clip(tp+fp, K.epsilon(), None)
            return precision
        elif metric == 'recall':
            recall = tp/K.clip(tp+fn, K.epsilon(), None)
            return recall
        elif metric == 'iou':
            iou = tp/K.clip(tp+fp+fn, K.epsilon(), None)
            return iou
        elif metric == 'f1':
            precision = tp/K.clip(tp+fp, K.epsilon(), None)
            recall = tp/K.clip(tp+fn, K.epsilon(), None)
            f1 = (2*precision*recall)/K.clip(precision+recall, K.epsilon(), None)
            return f1
        
    return metrics

def transform_data(path='./soccer_videos/'):
    leagues = ['AsianCup', 'WorldCup', 'PremierLeague', 'EuroCup', 'New_720p']
#     leagues = ['New_720p']
    
    event_files = []
    story_files = []
    shot_bound_files = []
    shot_type_files = []
    
    for league in leagues:
        league_event_path = path + 'event/' + league + '_event/'
        league_shot_path = path + 'shot/' + league + '_shot/'
        event_files.extend(glob.glob(league_event_path+'*event.txt'))
        story_files.extend(glob.glob(league_event_path+'*story.txt'))
        shot_bound_files.extend(glob.glob(league_shot_path+'*shotBoundary.txt'))
        shot_type_files.extend(glob.glob(league_shot_path+'*shotType.txt'))
                
    event_files = natsorted(event_files)
    story_files = natsorted(story_files)
    shot_bound_files = natsorted(shot_bound_files)
    shot_type_files = natsorted(shot_type_files)
            
    events_df = []
    story_df = []
    shot_bound_df = []
    shot_type_df = []
    
    for event_file, story_file, shot_bound_file, shot_type_file in zip(event_files, story_files, shot_bound_files, shot_type_files):
        df = pd.read_csv(event_file, delim_whitespace=True, index_col=None, names=['type','start', 'end', 'replay'])
        df['video'] = event_file.split('/')[-1].split('_')[0]
        df['league'] = event_file.split('_')[-1].split('.')[0]
        df['duration'] = (df['end'] - df['start'])/25
        events_df.append(df)
        
        df = pd.read_csv(story_file, delim_whitespace=True, index_col=None, names=['type','start', 'end', 'file'])
        df['video'] = story_file.split('/')[-1].split('_')[0]
        story_df.append(df)
        
        df = pd.read_csv(shot_bound_file, delim_whitespace=True, index_col=None, names=['start', 'end', 'file'])
        df['video'] = shot_bound_file.split('/')[-1].split('_')[0]
        shot_bound_df.append(df)
        
        df = pd.read_csv(shot_type_file, delim_whitespace=True, index_col=None, names=['start', 'end', 'type', 'file'])
        df['video'] = shot_type_file.split('/')[-1].split('_')[0]
        shot_type_df.append(df)

    events_df = pd.concat(events_df, axis=0).reset_index(drop=True)
    story_df = pd.concat(story_df, axis=0).reset_index(drop=True)
    shot_bound_df = pd.concat(shot_bound_df, axis=0).reset_index(drop=True)
    shot_type_df = pd.concat(shot_type_df, axis=0).reset_index(drop=True)

    return events_df, story_df, shot_bound_df, shot_type_df

def save_clips(df, videos_path, save_path, width=224, height=224, crop_lenght=4):
    df = df.sort_values(by=['video', 'start'])
    clip_filenames = df['video'].astype(str) + '_' + df['start'].astype(str) + '_' + df['end'].astype(str)
    qty = len(df.index)
    past_video = 0
    for i, ID in enumerate(df.index):
        filename = save_path + clip_filenames.loc[ID] + '.npy'
        print("%s/%s:"%(i,qty),filename)
        if os.path.isfile(filename) == True:
            print('Done')
        else:
            try:
                video_path = glob.glob(videos_path + '/*/*' + df.loc[ID]['video'] + '*.mp4')[0]
            except:
                print(video_path)
                print('Video not found!')
                continue
            if video_path != past_video:
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
            # start_frame = df.loc[ID]['start']
            end_frame = df.loc[ID]['end']
            start_frame = int(df.loc[ID]['end'] - crop_lenght*fps)

            selected_idx = np.linspace(start_frame, end_frame, 24).astype(int)
            selected_frames = np.empty(shape=(len(selected_idx), height, width, 3), dtype=np.uint8)
            i = 0
            for frame_idx in range(start_frame, end_frame+1):
                frame = cap.read()[1]
                if frame_idx in selected_idx:
                    selected_frames[i] = cv2.resize(frame, (width,height))
                    i += 1
            directory = os.path.abspath(save_path + clip_filenames.loc[ID])
            np.save(directory, selected_frames)
            past_video = video_path
            print('Done')
            
#def initialize_flownet2():
#    flownets = __import__("flownet2-pytorch.models")
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
#    parser.add_argument("--rgb_max", type=float, default=255.)
#    args, unknown = parser.parse_known_args()
#
#    net = flownets.models.FlowNet2(args).cuda()
#    dict = torch.load("./models/FlowNet2_checkpoint.pth.tar")
#    net.load_state_dict(dict["state_dict"])
#    return net

def create_optical_batch(net, x_batch):
    x_optical_batch_pure = np.zeros_like(x_batch, dtype=np.float)
    x_optical_batch = np.zeros_like(x_batch, dtype=np.uint8)
    for i in range(0, len(x_batch)-1):
        images = [cv2.resize(x_batch[i], (192,192)), cv2.resize(x_batch[i+1], (192,192))]
        images = np.array(images).transpose(3, 0, 1, 2)
        im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()
        result = net(im).squeeze()
        data = cv2.resize(result.data.cpu().numpy().transpose(1, 2, 0), (224,224))
        frame = flowlib.flow_to_image(data)
        x_optical_batch_pure[i+1,...,:2] = data
        x_optical_batch[i+1] = frame
    x_optical_batch[0] = x_optical_batch[1]
    return x_optical_batch, x_optical_batch_pure

def jupyter_animation(frames):
    plt.rcParams["animation.html"] = "jshtml"
    clips, widht, height, channels = frames.shape
    
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(10)
    h = ax.axis([0, widht, height,0])
    # plt.axis('off')
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    l = ax.imshow(np.ones((widht, height, 3), dtype=np.uint8))

    def animate(i):
        l.set_data(frames[i,...,::-1])

    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=clips)
    return ani

def window_nms(predictions, window=0):
    idxs = np.squeeze(np.argwhere(predictions == np.amax(predictions)))
    for i, distance in enumerate(np.ediff1d(idxs)):
        if distance <= window+1:
            predictions[idxs[i]:idxs[i+1]] = True
    return predictions

def updateJsonFile(json_path, result, start, end, start_frame, end_frame):
    jsonFile = open(json_path, "r")
    data = json.load(jsonFile)
    jsonFile.close()
    data['inferences'].append({
        'action': 'finishings',
        'probability': '%.2f'%result,
        'start_time': str(datetime.timedelta(seconds=int(start))),
        'end_time': str(datetime.timedelta(seconds=int(end))),
        'start_frame': int(start_frame),
        'end_frame': int(end_frame)})
    jsonFile = open(json_path, "w+")
    jsonFile.write(json.dumps(data))
    jsonFile.close()

# def game_inference(model, video_path, json_path, samples = 24, height = 224, width = 224, clip_duration=4):
#     full_path = os.path.abspath(video_path)
#     cap = cv2.VideoCapture(full_path)
#     video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     clip_length = int(fps*clip_duration)
#     video_name = video_path.split('data/')[-1]
#     data = {}
#     data['video'] = {
#         'path': full_path,
#         'fps': fps,
#         'number_frames': video_length,
#         'duration': str(datetime.timedelta(seconds=int(video_length/fps)))}
#     data['inferences'] = []
#     with open(json_path, 'w') as outfile:
#         json.dump(data, outfile)
#     selected_length = math.floor(((video_length - clip_length)*samples)/(clip_length))
#     start_frames = np.arange(0, selected_length, int(samples/4))
#     video = np.zeros(shape=(selected_length, height, width, 3), dtype=np.uint8)
#     selected_frames = np.linspace(0, video_length, selected_length, dtype='int')
#     aux = 0
#     first = True
#     for i in tqdm(range(video_length - 2*clip_length)):
#         if i in selected_frames:
#             video[aux] = cv2.resize(cap.read()[1], (width, height))
#             if aux in start_frames and aux>=samples:
#                 result = model.predict(np.expand_dims(video[aux-samples:aux]/255, 0))[0]
#                 if result[1] > 0.5:
#                     end_frame = i
#                     end_time = math.ceil(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
#                     start_frame = end_frame - clip_length
#                     start_time = end_time - clip_duration
#                     updateJsonFile(json_path, result[1], start_time, end_time, start_frame, end_frame)
# #                     np.save('./clips/%s_%i'%(os.path.basename(video_name),start_frame), video[aux-samples:aux])
#             aux += 1
#         else:
#             _ = cap.read()
#     return data

def game_inference(model, video_path, json_path, samples = 24, height = 224, width = 224, clip_duration=4):
    full_path = os.path.abspath(video_path)
    cap = cv2.VideoCapture(full_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    clip_length = int(fps*clip_duration)
    video_name = video_path.split('data/')[-1]
    data = {}
    data['video'] = {
        'path': full_path,
        'fps': fps,
        'number_frames': video_length,
        'duration': str(datetime.timedelta(seconds=int(video_length/fps)))}
    data['inferences'] = []
    with open(json_path, 'w') as outfile:
        json.dump(data, outfile)
    selected_frames = np.arange(0, video_length - 2*clip_length, (fps*clip_duration)/samples).astype(int)
    selected_length = len(selected_frames)
    start_frames = np.arange(0, selected_length, fps/2).astype(int)
    num_inf = 0
    video = np.zeros(shape=(selected_length, height, width, 3), dtype=np.uint8)
    aux = 0
    for i in tqdm(range(video_length)):
        if i in selected_frames:
            video[aux] = cv2.resize(cap.read()[1], (width, height))
            if aux in start_frames and aux>samples:
                result = model.predict(np.expand_dims(video[aux-samples:aux]/255, 0))[0]
                num_inf += 1
                if result[1] > 0.5:
                    end_frame = i-1
                    end_time = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                    start_frame = end_frame - clip_length
                    start_time = end_time - clip_duration
                    updateJsonFile(json_path, result[1], start_time, end_time, start_frame, end_frame)
            aux += 1
        else:
            _ = cap.read()
    print(num_inf)
    return data

def remove_duplicates_json(json_path):
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

    duplicated_idx = find_duplicates(start_times, window=1)

    for i, idx in enumerate(np.unique(duplicated_idx)):
        start = min(np.array(start_times)[duplicated_idx == idx])
        end = max(np.array(end_times)[duplicated_idx == idx])
        start_frame = min(np.array(start_frames)[duplicated_idx == idx])
        end_frame = max(np.array(end_frames)[duplicated_idx == idx])
        result = max(np.array(probabilities, dtype=np.float)[duplicated_idx == idx])
        updateJsonFile(new_json_path, result, start, end, start_frame, end_frame)
        
        
def save_clips_from_json(json_path, threshold=0.9):
    with open(json_path) as json_file:
        data = json.load(json_file)
    video_path = data['video']['path']
    filename = os.path.basename(video_path)
    folder = os.path.dirname(os.path.abspath(video_path))
    save_path = os.path.join(folder, filename.split('.')[0] + '_clips')
    if os.path.exists(save_path) == False:
        os.mkdir(save_path)
    for inference in data['inferences']:
        if float(inference['probability']) > threshold:
            clip_name = os.path.join(save_path, inference['start_time'] + '_' + inference['end_time'] + '.mkv')
            print(clip_name)
            subprocess.call(['ffmpeg', '-ss', inference['start_time'], '-to', inference['end_time'], '-i', video_path, '-map_metadata', '-1', '-c', 'copy', clip_name])

def find_duplicates(itens, window=1):
    idxs = np.arange(len(itens))
    for i, diff in enumerate(np.ediff1d(itens)):
        if diff > window:
            idxs[i+1] = idxs[i] + 1
        else:
            idxs[i+1] = idxs[i]
    return idxs
