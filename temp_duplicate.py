import glob

folders = glob.glob('./*')

for folder in folders:

    last_pose= sorted(glob.glob(folder + '/pose*'))[-1]
    print('Last pose: ', last_pose)

    higher_end = int(last_pose.split('_')[-1].split('.')[0])
    print('Highest index: ', higher_end)

    saliency_dummy = glob.glob(folder + '/saliency*')[0]
    # Rename dummy to saliency_00001.jpg

    for i in range(2, higher_end + 1):

        print(i)
        # cp dummy to saliency_{i}.jpg
