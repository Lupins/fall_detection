'''
Join frames together and save them as a video
'''
import os
import sys
import cv2
import numpy

POSITIVE_CLASS  = 'Fall'
NEGATIVE_CLASS  = 'NotFall'
OUT_EXTENSION   = '.avi'
WIDTH           = 224
HEIGHT          = 224
FPS             = 30

def iterate_class_folder(db_folder, in_extension, p_class = POSITIVE_CLASS, n_class = NEGATIVE_CLASS):

    # Positive class
    class_folder = db_folder + '/' + p_class + '/'
    v_folders = os.listdir(class_folder)
    for folder in v_folders:
        video_output = setup_video_output(class_folder, folder, in_extension)
        video_output = create_video_output(class_folder + folder + '/', folder,
                                           in_extension, video_output)
        video_output.release()

    # Negative class
    class_folder = db_folder + '/' + n_class + '/'
    v_folders = os.listdir(class_folder)
    for folder in v_folders:
        video_output = setup_video_output(class_folder, folder, in_extension)
        video_output = create_video_output(class_folder + folder + '/', folder,
                                           in_extension, video_output)
        video_output.release()

def setup_video_output(class_path, folder_name, in_extension, width = WIDTH,
                       height = HEIGHT, fps = FPS,
                       out_extension = OUT_EXTENSION):

    out_file = class_path + folder_name + '/' + folder_name + out_extension
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    v_out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))

    return v_out

def create_video_output(img_folder, img_name, in_extension, output):

    images = os.listdir(img_folder)

    for i in range(len(images)):
        frame_name = img_name + '-' + format(i, '04d') + in_extension
        frame_path = img_folder + frame_name

        frame = cv2.imread(frame_path)

        if frame is None:
            output.write(frame)

    return output

# def remove_extension_name(name):
    # return name.split('.')[0]

# def edit_video(file_name):

    # v_in = cv2.VideoCapture('input/' + file_name)
    # fourcc = cv2.VideoWriter_fourcc(*'MPEG')

    # width = int(v_in.get(3))
    # height = int(v_in.get(4))

    # output_name = remove_extension_name(file_name)
    # v_out = cv2.VideoWriter('output/' + output_name + '.avi',
                            # fourcc, v_in.get(5), (224, 224))

    # while(v_in.isOpened()):

        # flag, frame = v_in.read()

        # if flag:

            # c_frame = urfd_crop_depth_info(frame, width, height)
            # c_frame = resize_frame(c_frame, 224, 224)
            # c_frame = cv2.cvtColor(c_frame, cv2.COLOR_BGR2GRAY)
            # c_frame = cv2.cvtColor(c_frame, cv2.COLOR_GRAY2BGR)

            # print(type(c_frame))
            # print(c_frame.dtype)
            # print(c_frame.shape)
            # v_out.write(c_frame)

        # else:
            # break

    # v_in.release()
    # v_out.release()

def main(db_folder, in_extension):
    iterate_class_folder(db_folder, in_extension)

main(str(sys.argv[1]), str(sys.argv[2]))
