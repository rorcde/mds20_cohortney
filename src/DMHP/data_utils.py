import torch
import tarfile
import pickle
import pandas
from pathlib import Path
import numpy as np
import os
import re
import pandas as pd
import sys
sys.path.append("..")
def load_data(data_dir, maxsize=None, maxlen=-1, ext='txt', datetime=True, type_=None):
    """
    Loads the sequences saved in the given directory.

    Args:
        data_dir    (str, Path) - directory containing sequences
        maxsize     (int)       - maximum number of sequences to load
        maxlen      (int)       - maximum length of sequence, the sequences longer than maxlen will be truncated
        ext         (str)       - extension of files in data_dir directory
        datetime    (bool)      - variable meaning if time values in files are represented in datetime format

    Returns:
        ss          (List(torch.Tensor))    - list of torch.Tensor containing sequences. Each tensor has shape (L, 2) and represents event sequence 
                                                as sequence of pairs (t, c). t - time, c - event type.
        Ts          (torch.Tensor)          - tensor of right edges T_n of interavls (0, T_n) in which point processes realizations lie.
        class2idx   (Dict)                  - dict of event types and their indexes
        user_list   (List(Dict))            - representation of sequences siutable for Cohortny
             
    """
    s = []
    classes = set()
    nb_files = 0
    time_col = 'time'
    event_col = 'event'
    if ext == "tar.gz" or "pkl":
        if ext == "pkl":
            df = pd.read_pickle(Path(data_dir, "fx_data.pkl"))[:100000]
            for i in range (df.shape[0]):
                data = {'time': [df.iloc[i]['time']], 'event': [df.iloc[i]['ud']]}
                df_data = pd.DataFrame(data=data)
                classes = classes.union(set(df_data[event_col].unique()))
                s.append(df_data)
                print (f'Reading {i} out of {df.shape[0]}\n')
                
        if ext == "tar.gz":
            with tarfile.open(data_dir, "r:gz") as tar:
                fp = tar.extractfile("synthetic_hawkes_data")
                df = pickle.load(fp)
                fp.close()
            for i in range (len(df[3])):
                data = {'time': [df[3][i]], 'event': [df[4][i]]}
                df_data = pd.DataFrame(data=data)
                classes = classes.union(set(df_data[event_col].unique()))
                s.append(df_data)
                print (f'Reading {i} \n')
    if ext == "csv" or "txt":
        for file in sorted(os.listdir(data_dir), key=lambda x: int(re.sub(fr'.{ext}', '', x)) if re.sub(fr'.{ext}', '', x).isdigit() else 0):
            if file.endswith(f'.{ext}') and re.sub(fr'.{ext}', '', file).isnumeric():
                if maxsize is None or nb_files <= maxsize:
                    nb_files += 1
                else:
                    break


                if type_ == 'booking1':
                    time_col = 'checkin'
                    event_col = 'city_id'
                elif type_ == 'booking2':
                    time_col = 'checkout'
                    event_col = 'city_id'

                df = pd.read_csv(Path(data_dir, file))
                classes = classes.union(set(df[event_col].unique()))
                if datetime:
                    df[time_col] = pd.to_datetime(df[time_col])
                    df[time_col] = (df[time_col] - df[time_col][0]) / np.timedelta64(1,'D')
                if maxlen > 0:
                    df = df.iloc[:maxlen]
                s.append(df)
    classes = list(classes)
    class2idx = {clas: idx for idx, clas in enumerate(classes)}

    ss, Ts = [], []
    for i, df in enumerate(s):
        user_dict = dict()
        if s[i][time_col].to_numpy()[-1] < 0:
             continue
    
        s[i][event_col].replace(class2idx, inplace=True)
        for event_type in class2idx.values():
            dat = s[i][s[i][event_col] == event_type]
 
        st = np.vstack([s[i][time_col].to_numpy(), s[i][event_col].to_numpy()])
        tens = torch.FloatTensor(st.astype(np.float32)).T
      
        if maxlen > 0:
            tens = tens[:maxlen]
        ss.append(tens)
        Ts.append(tens[-1, 0])
    Ts = torch.FloatTensor(Ts)
    print ('Data processing completed')
    return ss, Ts, class2idx

#transforming data to the array taking into account an event type
def sep_hawkes_proc(user_list, event_type):
    sep_seqs = []
    for user_dict in user_list:
        sep_seqs.append(np.array(user_dict[event_type], dtype = np.float32))

    return sep_seqs
