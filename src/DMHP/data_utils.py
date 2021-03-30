import torch
import tarfile
import pickle
import pandas
from pathlib import Path
import numpy as np
import os
import re
import json
import pandas as pd
import sys
sys.path.append("..")
def load_data(data_dir):
    """
    Loads the sequences saved in the given directory.

    Args:
        data_dir    (str, Path) - directory containing sequences


    Returns:
        ss          (List(torch.Tensor))    - list of torch.Tensor containing sequences. Each tensor has shape (L, 2) and represents event sequence 
                                                as sequence of pairs (t, c). t - time, c - event type.
        Ts          (torch.Tensor)          - tensor of right edges T_n of interavls (0, T_n) in which point processes realizations lie.
        class2idx   (Dict)                  - dict of event types and their indexes
        user_list   (List(Dict))            - representation of sequences siutable for Cohortny
             
    """
  
    with open(Path(data_dir, 'info.json')) as info:
        info = json.load(info)
    classes = info['classes']
    seq_nmb = info['seq_nmb']
    gt_ids = None
    if Path(data_dir, 'clusters.csv').exists():
        gt_ids = pd.read_csv(Path(data_dir, 'clusters.csv'))['cluster_id'].to_numpy()
        gt_ids = torch.LongTensor(gt_ids)
    nb_files = 0
    time_col = 'time'
    event_col = 'event'
    class2idx = {clas: idx for idx, clas in enumerate(classes)}

    ss, Ts = [], []
    ext = 'csv'
    for file in sorted(os.listdir(data_dir), key=lambda x: int(re.sub(fr'.{ext}', '', x)) if re.sub(fr'.{ext}', '', x).isdigit() else 0):
        if file.endswith(f'.{ext}') and re.sub(fr'.{ext}', '', file).isnumeric():
          user_dict = dict()
          f = pd.read_csv(Path(data_dir, file))
          if f[time_col].to_numpy()[-1] < 0:
                 continue

          f[event_col].replace(class2idx, inplace=True)
          for event_type in class2idx.values():
              dat = f[f[event_col] == event_type]


          st = np.vstack([f[time_col].to_numpy(), f[event_col].to_numpy()])
          tens = torch.FloatTensor(st.astype(np.float32)).T
          ss.append(tens)
          Ts.append(tens[-1, 0])
    Ts = torch.FloatTensor(Ts)
    print ('Data processing completed')
    return ss, Ts, class2idx, gt_ids
