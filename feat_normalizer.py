'''
this script reads in the feature table before vectorizing, and normalize all numerical features from 0 to 1
'''
import pandas as pd
import numpy as np


def feat_norm():
    black_list = ['ip', 'class', 'tld']

    feat_table = pd.read_csv('./tmp/features.txt', delimiter='\t')

    header = list(feat_table.columns)
    feat_matrix = pd.DataFrame()
    train_norm_args = []
    for i in header:
        if i in black_list:
            feat_matrix[i] = feat_table.ix[:, i]
        else:
            line = feat_table.ix[:, i]
            mean_ = line.mean()
            max_ = line.max()
            min_ = line.min()
            feat_matrix[i] = (line - mean_) / (max_ - min_)
            train_norm_args.append((mean_, max_, min_))
        # print('converted %s' % i)
    return train_norm_args, feat_matrix


def main():
    _, feat_matrix = feat_norm()
    feat_matrix.to_csv('./tmp/features_norm.txt')


if __name__ == '__main__':
    main()
