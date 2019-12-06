import argparse as ap
import numpy as np
import cv2
import glob

DATA = None
CLASS = None
EXT = None
class Npy:
    def __init__(self, dataset, classes, streams, ext):
        self.dataset = dataset

        self.classes = classes
        print(type(streams), streams)
        self.streams = streams
        self.ext = ext

        return


    def process(self):
        self.process_classes()

    def process_classes(self):

        for _class in self.classes:
            for _stream in self.streams:
                # path = self.dataset + _class + '/*/' + _stream + '*'
                path = self.dataset + _class + '/*'

                print(path)
                # frames = sorted(glob.glob(path))
                folders = sorted(glob.glob(path))
                # self.loop_frame(frames, _stream)
                for folder in folders:
                    # print('Folder', folder)
                    frames = sorted(glob.glob(folder + '/' + _stream + '*'))
                    self.loop_frame(frames, _stream)


    def loop_frame(self, frames, stream):

        idx = 0
        stack_idx = 0
        stack = np.zeros((24, 224, 224, 3))
        for frame in frames:
            
            img = cv2.imread(frame)
            stack[idx] = img
            idx += 1

            if idx == 24:
                npy_path = frame.rsplit('/', 1)[0]
                print(npy_path + '/' + stream + '_' + str('{0:0=4d}'.format(stack_idx)) + '.npy')
                np.save(npy_path + '/' + stream + '_' + str('{0:0=3d}'.format(stack_idx)), stack)
                stack_idx += 1
                idx = 0
                stack = np.zeros((24, 224, 224, 3))


def argument_setup():
    parser = ap.ArgumentParser(description='Create .npy from videos')
    parser.add_argument('-dataset', type=str, help='Dataset folder')
    parser.add_argument('-classes', type=str, nargs='*', help='Classes of the dataset')
    parser.add_argument('-streams', type=str, nargs='*', help='Streams to use')
    parser.add_argument('-ext', type=str, help='Extension of the video file')

    return parser.parse_args()


args = argument_setup()
npy = Npy(args.dataset, args.classes, args.streams, args.ext)

npy.process()
