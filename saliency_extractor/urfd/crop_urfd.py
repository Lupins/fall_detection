import os
import sys
import cv2

# Output folder
OUT_PATH = 'urfd_cropped/'

# Iterate over every file in the entry folder
def iterate_over_folder(folder, extension):
    files = os.listdir(folder)

    for file in files:

        # Ignores any file with a diferent extenstion than 'extension'
        if file.split('.')[1] != extension:
            continue

        crop_video(folder + file)

# Reads all video's frames, crops them and prepares the output file name
def crop_video(video):

    v_in = cv2.VideoCapture(video)

    width = int(v_in.get(3))
    height = int(v_in.get(4))
    n_frames = int(v_in.get(7))

    print(width, height, n_frames)

    file_name = remove_extension_name(video)
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    v_out = cv2.VideoWriter(OUT_PATH + file_name + '.avi', fourcc, v_in.get(5),
                            (int(width / 2), height))

    while v_in.isOpened():

        flag, frame = v_in.read()

        if flag:
            frame = crop_image_width_half(frame, width, height)

            v_out.write(frame)

        else:
            break

    v_in.release()
    v_out.release()

# Remove the nested path and extension from the entry file name
def remove_extension_name(str):
    print(str)
    str = str.split('.')[0]
    str = str.split('/')[1]
    return str

# Exclusive to URFD, crops the image in half, removing the depth information
def crop_image_width_half(img, width, height):
    return img[0:height, int(width / 2):width]

# Main
def main(folder, extension):

    # Creates the output folder
    if not os.path.exists(OUT_PATH):
        try:
            os.mkdir(OUT_PATH)
        except OSError:
            print('Failed to create', OUT_PATH)
        else:
            print('Succeed to created', OUT_PATH)

    iterate_over_folder(folder, extension)

# Start here
main(str(sys.argv[1]), str(sys.argv[2]))
