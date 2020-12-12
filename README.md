# Models of Sequential Data Project: COHORTNEY

The project is about implementation of methods for event sequence clustering.

The main method to be considered: COHORTNEY

The main baseline is [A Dirichlet Mixture Model of Hawkes Processes for
Event Sequence Clustering](https://arxiv.org/pdf/1701.09177.pdf)

Datasets:
- IPTV dataset
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

in root dir:

```bat
python -m venv cohortney

. cohortney/bin/activate

cd mds20_cohortney

pip install -r requirements.txt
```

The code organization structure:
