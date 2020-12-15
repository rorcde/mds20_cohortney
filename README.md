# Models of Sequential Data Project: COHORTNEY

The project is about implementation of methods for event sequence clustering.

The main method to be considered: COHORTNEY

The main baseline is [A Dirichlet Mixture Model of Hawkes Processes for
Event Sequence Clustering](https://arxiv.org/pdf/1701.09177.pdf)

Datasets:
- IPTV dataset
- Synthetic Hawkes processes
- other datasets from the PointProcesses.com

Tools:
- Pytorch lighting
- TQDM
- Pydata stack library

### Installation

```bat
git clone https://github.com/rodrigorivera/mds20_cohortney
```

### Setup


```bat
cd [VENV]

virtualenv cohortney

source cohortney/bin/activate
```

back in repository dir:
```bat
pip install -r requirements.txt

python setup.py install
```

The code organization structure:


