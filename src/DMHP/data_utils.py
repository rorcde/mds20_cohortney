import pandas as pd
from pathlib import Path
import numpy as np
import torch


def load_data(path, nfiles, maxlen=-1, endtime=None, datetime=False, ext='csv'):
    s = []
    classes = set()
    for i in range(1, nfiles + 1):
        path_ = Path(path, f'{i}.{ext}')
        with path_.open('r') as f:
            df = pd.read_csv(f)
        classes = classes.union(set(df['event'].unique()))
        if datetime:
            df['time'] = pd.to_datetime(df['time'])
            df['time'] = (df['time'] - df['time'][0]) / np.timedelta64(1,'D')
        if maxlen > 0:
            df = df.iloc[:maxlen]
        s.append(df[['time', 'event']])

    classes = list(classes)
    class2idx = {clas: idx for idx, clas in enumerate(classes)}

    ss, Ts = [], []
    for i, df in enumerate(s):
        if s[i]['time'].to_numpy()[-1] < 0:
            continue
        s[i]['event'].replace(class2idx, inplace=True)
        tens = torch.FloatTensor(np.vstack([s[i]['time'].to_numpy(), s[i]['event'].to_numpy()])).T
        if maxlen > 0:
            tens = tens[:maxlen]
        ss.append(tens)
        if endtime is not None:
            Ts.append(endtime)
        else:
            Ts.append(tens[-1, 0] * 1.01)

    Ts = torch.FloatTensor(Ts)

    return ss, Ts, class2idx
    