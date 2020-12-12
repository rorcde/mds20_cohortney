import torch
import pandas
from pathlib import Path
import numpy as np
import os
import re
import pandas as pd


def load_data(data_dir, maxlen=-1, ext='txt', datetime=True):
    s = []
    classes = set()
    for file in os.listdir(data_dir):
        if file.endswith(f'.{ext}') and re.sub(fr'.{ext}', '', file).isnumeric():
            df = pd.read_csv(Path(data_dir, file))
            classes = classes.union(set(df['event'].unique()))
            df['time'] = pd.to_datetime(df['time'])
            if datetime:
                df['time'] = (df['time'] - df['time'][0]) / np.timedelta64(1,'D')
            if maxlen > 0:
                df = df.iloc[:maxlen]
            df = df.drop(['id', 'option1'], axis=1)
            s.append(df)

    classes = list(classes)
    class2idx = {clas: idx for idx, clas in enumerate(classes)}

    ss, Ts = [], []
    user_list = []
    for i, df in enumerate(s):
      user_dict = dict()
      if s[i]['time'].to_numpy()[-1] < 0:
             continue
      s[i]['event'].replace(class2idx, inplace=True)
      for  event_type in class2idx.values():
          dat = s[i][s[i]['event'] == event_type]
          user_dict[event_type] = dat['time'].to_numpy()
      user_list.append(user_dict)


      st = np.vstack([s[i]['time'].to_numpy(), s[i]['event'].to_numpy()])
      tens = torch.FloatTensor(st.astype(np.float32)).T
      
      if maxlen > 0:
          tens = tens[:maxlen]
      ss.append(tens)
      Ts.append(tens[-1, 0])

    Ts = torch.FloatTensor(Ts)

    return ss, Ts, class2idx, user_list 


#transforming data to the array taking into account an event type
def sep_hawkes_proc(user_list, event_type):
    sep_seqs = []
    for user_dict in user_list:
        sep_seqs.append(np.array(user_dict[event_type], dtype = np.float32))

    return sep_seqs
