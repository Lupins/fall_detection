'''
I extracted the saliency from the URFD dataset ignoring the fact that every fall
video also has a not fall sequency, and the network might learn the wrong
patterns from them. The solution is to download the URFD .csv containing each
frame notation and use it to split the fall videos into actually fall and ADL.
The ADL's frames from the fall video is them moved to the NotFalls folder, with
a simple name, a _0X is attached to the end of the folder, X = 0 means it
happend before the fall sequence and X = 1 or greater, not the case, means it
happend after the fall sequence.
'''
import csv
import os

C_FOLDER = 'Falls/'
N_FOLDER = 'NotFalls/'
CSV_L = []

with open('hot_fix.csv', newline='') as csv_f:
    reader = csv.reader(csv_f, delimiter=' ', quotechar='|')
    for row in reader:
        row = str(row[0]).split(',')
        CSV_L.append(row)

p_folder = None
c_folder = None
i_state = -1
p_state = 0
c_state = None

for i in range(len(CSV_L)):

    c_folder = CSV_L[i][0]

    if c_folder != p_folder:
        p_state = 0
        i_state = -1

    c_state = int(CSV_L[i][2])

    # print(CSV_L[i][0], CSV_L[i][1], CSV_L[i][2])
    # Move to ADL
    if int(CSV_L[i][2]) != 0:
        if c_state != p_state:
            i_state += 1

        try:
            os.mkdir(N_FOLDER + CSV_L[i][0] + '-cam0_0' + str(i_state) + '/')
            None
        except:
            None

        os.rename(C_FOLDER + CSV_L[i][0] + '-cam0/' + 'saliency_' + str(int(CSV_L[i][1]) - 1).zfill(4) + '.png',
                  N_FOLDER + CSV_L[i][0] + '-cam0_0' + str(i_state) + '/' + 'saliency_' + str(int(CSV_L[i][1]) - 1).zfill(4) + '.png')

        print(C_FOLDER + CSV_L[i][0] + '-cam0/' + 'saliency_' + str(int(CSV_L[i][1]) - 1).zfill(4) + '.png\t' + N_FOLDER + CSV_L[i][0] + '-cam0_0' + str(i_state) + '/' + 'saliency_' + str(int(CSV_L[i][1]) - 1).zfill(4) + '.png')

    p_state = c_state
    p_folder = c_folder

