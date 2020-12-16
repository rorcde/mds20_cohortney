# not used in final
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import os
import re
from pathlib import Path
import pandas as pd
import numpy as np


class EventSequenceDataset(Dataset):
    def __init__(self, data_dir, ext='txt', datetime=True, maxlen=-1):
        self.dataset = self.load_dataset(data_dir, ext, datetime, maxlen)
        self.len = len(self.dataset)

    @staticmethod
    def load_dataset(data_dir, ext='txt', datetime=True, maxlen=-1):
        seqs = []

        for file in os.listdir(data_dir):
            if file.endswith(f'.{ext}') and re.sub(fr'.{ext}', '', file).isnumeric():
                df = pd.read_csv(Path(file))
                classes = classes.union(set(df['event'].unique()))
                if datetime:
                    df['time'] = pd.to_datetime(df['time'])
                    df['time'] = (df['time'] - df['time'][0]) / np.timedelta64(1,'D')
                if maxlen > 0:
                    df = df.iloc[:maxlen]
                seqs.append(df[['time', 'event']])

        classes = list(classes)
        class2idx = {clas: idx for idx, clas in enumerate(classes)}

        dataset = []
        for df in seqs:
            if df['time'].to_numpy()[-1] < 0:
                continue
            df['event'].replace(class2idx, inplace=True)
            tens = torch.FloatTensor(np.vstack([df['time'].to_numpy(), df['event'].to_numpy()])).T
            if maxlen > 0:
                tens = tens[:maxlen]
            dataset.append(tens)

        return dataset, class2idx
        
    def __getitem__(self, index):
        seq = self.dataser[index]
        
        return seq

    def __len__(self):
        return self.len

    
def pad_collate1(batch):
    lens = [len(x) for x in batch]
    batch_pad = pad_sequence(batch, batch_first=True, padding_value=0)

    return batch_pad, lens


def pad_collate2(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)

    return xx_pad, lens, yy
