import torch
import tarfile
import pickle
import pandas
import json
import argparse
from pathlib import Path
import numpy as np
import shutil
from shutil import copyfile
import os
import re
import pandas as pd
import sys
from numpy import asarray
from numpy import savetxt
sys.path.append("..")
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='dir holding sequences as separate files')
    parser.add_argument('--maxlen', type=int, default=500, help='maximum length of sequence')
    parser.add_argument('--ext', type=str, default='tar.gz', help='extention of files with sequences')
    parser.add_argument('--datetime', type=bool, default=False, help='if time values in event sequences are represented in datetime format')
    parser.add_argument('--save_dir', type=str, default = './', help='path to save results')
    parser.add_argument('--maxsize', type=int, default=None, help='max number of sequences')
    args = parser.parse_args()
    return args
def tranform_data(args):
    """
    Loads the sequences saved in the given directory.
    Args:
        data_dir    (str, Path) - directory containing sequences
        save_dir - directory for saving transform data
        maxsize     (int)       - maximum number of sequences to load
        maxlen      (int)       - maximum length of sequence, the sequences longer than maxlen will be truncated
        ext         (str)       - extension of files in data_dir directory
        datetime    (bool)      - variable meaning if time values in files are represented in datetime format
             
    """
    data_dir = args.data_dir
    save_dir = args.save_dir
    os.makedirs(save_dir)
    maxsize = args.maxsize
    maxlen = args.maxlen  
    ext = args.ext
    datetime = args.datetime
    classes = set()
    nb_files = 0
    time_col = 'time'
    event_col = 'event'
    gt_ids = None
    if args.ext == "pkl":
        with open(Path(args.data_dir, "fx_labels"), "rb") as fp:
            gt_ids = pickle.load(fp)[:maxsize]
            labels = np.unique(gt_ids)
            gt_data = []
            for i in range (len(gt_ids)):
                gt_data.append(int(np.nonzero(gt_ids[i] == labels)[0]))
            gt = {'cluster_id': gt_data}
            print(gt_data)
            gt_table = pd.DataFrame(data=gt)
            gt_table.to_csv(Path(save_dir, 'clusters.csv'))
    if Path(args.data_dir, 'clusters.csv').exists():
        gt_ids = pd.read_csv(Path(args.data_dir, 'clusters.csv'))[:(maxsize)]
        gt_ids.to_csv(Path(save_dir, 'clusters.csv'))
    



args = parse_arguments()
print(args)
tranform_data(args)