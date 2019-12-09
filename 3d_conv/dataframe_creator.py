import argparse as ap
import os
import glob
import pandas as pd


def argument_setup():
    parser = ap.ArgumentParser(description='Create dataframes for .npy files')
    parser.add_argument('-output', type=str, help='Output folder (with /) of the .csv')
    parser.add_argument('-section', type=str, help='Whether this dataframe is TRAIN, VALIDATION, or TEST')
    parser.add_argument('-dataset', type=str, help='Dataset path (with /)')
    parser.add_argument('-classes', type=str, nargs='*', help='Classes of the dataset')

    return parser.parse_args()


args = argument_setup()

data = []

for c_class in args.classes:
    for row in sorted(glob.glob(args.dataset + c_class + '/**/*.npy')):
        # print(row

        video_name = row.split('/')[-2:-1]
        stream = row.split('/')[-1].split('_')[0]

        npy_idx = row.split('/')[-1].split('_')[-1].split('.')[0]

        i_class = -1
        if c_class == 'Falls':
            i_class = 0
        else:
            i_class = 1

        data.append([i_class, video_name, stream, npy_idx, row])

df = pd.DataFrame(data, columns=['class', 'video_name', 'stream', 'npy_idx', 'path'])
df.to_csv(args.output + args.section + '.csv', ',')
