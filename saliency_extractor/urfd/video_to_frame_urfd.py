import os
import sys
import cv2

# Output folder
OUT_PATH = 'urfd_frames/'

# Iterate over every file in the entry folder
def iterate_over_folder(folder, extension):
    files = os.listdir(folder)

    for file in files:
        print(file)

        # Ignores any file with a diferent extenstion than 'extension'
        if file.split('.')[1] != extension:
            continue

        extract_frames(folder, file)

# Save each frame of the video individually
def extract_frames(path, video):
    v_in = cv2.VideoCapture(path + video)

    i = 0
    while v_in.isOpened():

        flag, frame = v_in.read()

        if flag:

            file_name = form_file_name(video, i)
            cv2.imwrite(OUT_PATH + file_name + '.png', frame)

            i += 1
        else:
            break

    v_in.release()

# Create the proper output file name
def form_file_name(str, i_frame):
    str = str.split('.')[0]
    str = str + '-' + format(i_frame, '04d')
    return str

# Remove the nested path and extension from the entry file name
def remove_extension_name(str):
    print(str)
    str = str.split('.')[0]
    str = str.split('/')[1]
    return str

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
