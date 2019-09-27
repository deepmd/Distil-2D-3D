from __future__ import print_function, division
import os
import sys
import pandas as pd


def concat_hmdb51_csvs(csv_dir_path, split_index):
    dataset = {'train': [], 'test': []}
    labels = []
    for filename in sorted(os.listdir(csv_dir_path)):
        if 'split{}'.format(split_index) not in filename:
            continue

        label = '_'.join(filename.split('_')[:-2])
        labels.append(label)

        data = pd.read_csv(os.path.join(csv_dir_path, filename),
                           delimiter=' ', header=None)
        videonames = []
        splits = []
        for i in range(data.shape[0]):
            row = data.ix[i, :]
            if row[1] == 0:
                continue
            elif row[1] == 1:
                split = 'train'
            elif row[1] == 2:
                split = 'test'

            videonames.append(row[0])
            splits.append(split)

        for i in range(len(videonames)):
            videoname = videonames[i]
            dataset[splits[i]].append(label + '/' + videoname)

    for split_name, videos in dataset.items():
        dst_path = os.path.join(csv_dir_path, '{}list{:02d}.txt'.format(split_name, split_index))
        df = pd.DataFrame(videos)
        df.to_csv(dst_path, index=False, header=False)

    dst_path = os.path.join(csv_dir_path, 'classInd.txt')
    df = pd.DataFrame(labels)
    df.index = df.index + 1
    df.to_csv(dst_path, sep=' ', header=False)


if __name__ == '__main__':
    csv_dir_path = sys.argv[1]

    for split_index in range(1, 4):
        concat_hmdb51_csvs(csv_dir_path, split_index)
