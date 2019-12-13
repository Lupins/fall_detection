import os
import sys
import glob2
import shutil

# argv[1] = origin folder
# argv[2] = dataset dest

# Move all the saliency from argv[1] folder to their proper folder in argv[2]

origin = sys.argv[1] + '/'
c_folder = origin.split('/')[-2] + '/'
dest_folder = sys.argv[2] + c_folder

saliency_files = sorted(glob2.glob(origin + 'saliency*'))

print('\nFolder: ', c_folder)

for file in saliency_files:

    print()
    f_number = file.split('/')[-1].split('_')[-1].split('.')[0]
    frame_name = 'frame_' + f_number + '.jpg'

    if os.path.isfile(file):
        try:
            os.remove(origin + frame_name)
            print('Delete: ', origin + frame_name)
        except:
            print()

    try:
        shutil.move(file, dest_folder)
        print('Move')
        print('From: ', file)
        print('To: ', dest_folder)
    except:
        print()


