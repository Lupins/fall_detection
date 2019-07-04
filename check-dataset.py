import os
import sys

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

            # print(video)
            # video001_frame_001
            frames = os.listdir(path + '/' + _class + '/' + video)
            frames.sort()
            print(video + ': ' + str(len(frames)))
            for i in range(1, len(frames)+1):
                j = "{:03d}".format(i)
                for frame in frames:
                    if j in frame:
                        print(frame)

main(sys.argv[1])
