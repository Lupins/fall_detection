import os
import sys
import glob

def main(path):

    # Falls NotFalls
    classes = os.listdir(path)
    classes.sort()
    for _class in classes:

        print(_class)
        # Folder: video001, video002
        videos = os.listdir(path + '/' + _class)
        videos.sort()
        for video in videos:

            frames = glob.glob(path + '/' + _class + '/' + video + '/frame*')
            s_frames = glob.glob(path + '/' + _class + '/' + video + '/saliency*')
            frames.sort(reverse=True)
            s_frames.sort(reverse=True)

            for idx in range(len(frames)):
                frame = frames[idx]

                frame_idx = frame.split('_')[-1]
                frame_idx = frame_idx.split('.')[0]

                # print(frame.split('/')[-1], end='\t')
                flag = False
                for s_frame in frames:
                    if frame_idx in s_frame:
                        # print(s_frame.split('/')[-1])
                        flag = True

                if flag == False:
                    print(frame.split('/')[-1], '\t*')


            # for idx in range(len(s_frames)):
                # frame = frames[idx]
                # s_frame = s_frames[idx]

                # print('From: ' + s_frame)
                # frame_idx = s_frame.split('_')[-1]
                # frame_idx = frame_idx.split('.')[0]
                # frame_idx = int(frame_idx)

                # print('To: ' + path + '/' + _class + '/' + video + '/saliency_' + ("{:05d}".format(len(s_frames)-idx)) + '.jpg')
                # os.rename(s_frame, path + '/' + _class + '/' + video + '/saliency_' + ("{:05d}".format(len(s_frames)-idx)) + '.jpg')

main(sys.argv[1])
