import cv2
import numpy

def urfd_crop_depth_info(frame, width, height):
    return frame[0:height, int(width / 2):width]

def resize_frame(frame, width, height):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

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

edit_video('fall-07-cam0.mp4')
