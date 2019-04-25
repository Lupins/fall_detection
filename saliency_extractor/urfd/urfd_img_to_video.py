'''
Join frames together and save them as a video
'''
import os
import sys
import cv2
import numpy

WIDTH = 224
HEIGHT = 224
FPS = 30

def remove_extension_name(name):
    return name.split('.')[0]

def edit_video(file_name):

    v_in = cv2.VideoCapture('input/' + file_name)
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')

    width = int(v_in.get(3))
    height = int(v_in.get(4))

    output_name = remove_extension_name(file_name)
    v_out = cv2.VideoWriter('output/' + output_name + '.avi',
                            fourcc, v_in.get(5), (224, 224))

    while(v_in.isOpened()):

        flag, frame = v_in.read()

        if flag:

            c_frame = urfd_crop_depth_info(frame, width, height)
            c_frame = resize_frame(c_frame, 224, 224)
            c_frame = cv2.cvtColor(c_frame, cv2.COLOR_BGR2GRAY)
            c_frame = cv2.cvtColor(c_frame, cv2.COLOR_GRAY2BGR)

            print(type(c_frame))
            print(c_frame.dtype)
            print(c_frame.shape)
            v_out.write(c_frame)

        else:
            break

    v_in.release()
    v_out.release()

def iterate_urfd_folder(in_folder, out_folder, extension):

    folders = os.listdir(in_folder)
    for folder in folders:
        create_video_from_frames(in_folder, folder, out_folder, extension)
        # break

def create_video_from_frames(path, folder, out_folder, extension, width = WIDTH,
                             height = HEIGHT, fps = FPS):

    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    v_out = cv2.VideoWriter(out_folder + '/' + folder + '.avi',
                            fourcc, fps, (width, height))

    v_out = iterate_images_folder(v_out, path, folder, extension)
    v_out.release()

def iterate_images_folder(output, path, folder, extension):

    images = os.listdir(path + '/' + folder)

    for i in range(len(images)):
        frame_name = folder + '-' + format(i, '04d') + extension
        path_frame = path + '/' + folder + '/' + frame_name

        frame = cv2.imread(path_frame)

        output.write(frame)

    return output


def main(img_folder, out_folder, extension):
    iterate_urfd_folder(img_folder, out_folder, extension)

main(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]))
